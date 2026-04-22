use std::collections::{BTreeSet, HashSet};
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use api::{
    resolve_api_key as resolve_nanogpt_api_key, resolve_api_key_for, resolve_base_url_for,
    save_openai_codex_credentials, ApiError, ApiService, ContentBlockDelta, ImageSource,
    InputContentBlock, InputMessage, MessageRequest, MessageResponse, NanoGptClient,
    OpenAiCodexCredentials, OutputContentBlock, ReasoningEffort, StreamEvent as ApiStreamEvent,
    ThinkingConfig, ToolChoice, ToolDefinition, ToolResultContentBlock, OPENAI_CODEX_CLIENT_ID,
    OPENAI_CODEX_ISSUER,
};
use crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers,
};
use crossterm::execute;
use crossterm::style::{Color, Stylize};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use platform::pebble_config_home as resolve_pebble_config_home;
use plugins::{PluginError, PluginManager, PluginSummary};
use reqwest::blocking::Client as BlockingClient;
use reqwest::header::{HeaderName, HeaderValue};
use serde::Deserialize;
use serde_json::Value as JsonValue;
use sha2::{Digest, Sha256};

use crate::init::{initialize_repo, initialize_repo_with_pebble_md, render_init_pebble_md};
use crate::input;
use crate::models::{
    context_length_for_model, current_service_or_default, default_model_or,
    infer_service_for_model, load_model_state, max_output_tokens_for_model_or, open_model_picker,
    open_provider_picker, persist_current_model, persist_provider_for_model,
    persist_proxy_tool_calls, provider_for_model, proxy_tool_calls_enabled,
    validate_provider_for_model,
};
use crate::proxy::{
    build_proxy_system_prompt, convert_messages_for_proxy, parse_proxy_response, parse_proxy_value,
    ProxyCommand, ProxyMessage, ProxySegment, RuntimeToolSpec,
};
use crate::render::{MarkdownStreamState, Spinner, TerminalRenderer};
use crate::ui;
use commands::{
    command_names_and_aliases, handle_agents_slash_command, handle_branch_slash_command,
    handle_skills_slash_command, handle_worktree_slash_command, render_help_topics_overview,
    render_slash_command_help, render_slash_command_help_topic, SlashCommand,
};
use compat_harness::{extract_manifest, UpstreamPaths};
use runtime::{
    auto_compaction_threshold_from_env, load_system_prompt_with_model_family, mcp_tool_name,
    resolve_sandbox_status, spawn_mcp_stdio_process, ApiClient, ApiRequest, AssistantEvent,
    CompactionConfig, ConfigLoader, ConfigSource, ContentBlock, ConversationMessage,
    ConversationRuntime, JsonRpcId, JsonRpcRequest, JsonRpcResponse, McpClientAuth,
    McpClientBootstrap, McpClientTransport, McpInitializeClientInfo, McpInitializeParams,
    McpListToolsParams, McpListToolsResult, McpToolCallParams, McpToolCallResult, McpTransport,
    MessageRole, PermissionMode, PermissionPolicy, PermissionPromptDecision, PermissionPrompter,
    PermissionRequest, RuntimeError, ScopedMcpServerConfig, Session, SessionMetadata, TokenUsage,
    ToolError, ToolExecutor, UsageTracker,
};
use tools::{
    build_plugin_manager, current_tool_registry, set_active_backend_service, GlobalToolRegistry,
};

const DEFAULT_MODEL: &str = "zai-org/glm-5.1";
const DEFAULT_MAX_TOKENS: u32 = 4096;
const DEFAULT_THINKING_BUDGET_TOKENS: u32 = 2_048;
const INIT_PEBBLE_MD_MAX_TOKENS: u32 = 2_048;
const DEFAULT_DATE: &str = "2026-03-31";
const SECRET_PROMPT_STALE_ENTER_WINDOW: Duration = Duration::from_millis(150);
const OPENAI_CODEX_DEVICE_AUTH_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const OPENAI_CODEX_DEVICE_POLL_SAFETY_MARGIN: Duration = Duration::from_secs(3);
// Retained for the structured proxy paths and external callers that still
// want to truncate large payloads. The TUI now uses tighter per-tool limits
// inline (see `render_bash_preview` etc.).
#[allow(dead_code)]
const MAX_TOOL_PREVIEW_CHARS: usize = 4_000;
#[allow(dead_code)]
const MAX_TOOL_PREVIEW_LINES: usize = 48;
const MAX_INIT_CONTEXT_CHARS: usize = 1_200;
const MAX_INIT_CONTEXT_FILES: usize = 6;
const MAX_INIT_TOP_LEVEL_ENTRIES: usize = 40;
const MCP_DISCOVERY_TIMEOUT_SECS: u64 = 30;
const AUTO_COMPACTION_CONTEXT_UTILIZATION_PERCENT: u64 = 85;
const AUTO_COMPACTION_CONTEXT_SAFETY_MARGIN_TOKENS: u64 = 8_192;
const VERSION: &str = env!("CARGO_PKG_VERSION");
const BUILD_TARGET: Option<&str> = option_env!("PEBBLE_BUILD_TARGET");
const IMAGE_REF_PREFIX: &str = "@";
const SELF_UPDATE_REPOSITORY: &str = "nanogpt-community/pebble";
const SELF_UPDATE_LATEST_RELEASE_URL: &str =
    "https://api.github.com/repos/nanogpt-community/pebble/releases/latest";
const SELF_UPDATE_USER_AGENT: &str = "pebble-self-update";
const OLD_SESSION_COMPACTION_AGE_SECS: u64 = 60 * 60 * 24;
const CHECKSUM_ASSET_CANDIDATES: &[&str] = &[
    "SHA256SUMS",
    "SHA256SUMS.txt",
    "sha256sums",
    "sha256sums.txt",
    "checksums.txt",
    "checksums.sha256",
];

type AllowedToolSet = BTreeSet<String>;

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    match parse_args(&args)? {
        CliAction::DumpManifests => dump_manifests(),
        CliAction::BootstrapPlan => print_bootstrap_plan(),
        CliAction::PrintSystemPrompt { cwd, date } => print_system_prompt(cwd, date),
        CliAction::Model { model } => handle_model_action(model)?,
        CliAction::Provider { provider } => handle_provider_action(provider)?,
        CliAction::Proxy { mode } => handle_proxy_action(mode)?,
        CliAction::Mcp { action } => handle_mcp_action(action)?,
        CliAction::Plugins { action, target } => {
            println!(
                "{}",
                handle_plugins_command(action.as_deref(), target.as_deref())?
            )
        }
        CliAction::Branch { action, target } => println!(
            "{}",
            handle_branch_slash_command(
                action.as_deref(),
                target.as_deref(),
                &env::current_dir()?
            )?
        ),
        CliAction::Worktree {
            action,
            path,
            branch,
        } => println!(
            "{}",
            handle_worktree_slash_command(
                action.as_deref(),
                path.as_deref(),
                branch.as_deref(),
                &env::current_dir()?,
            )?
        ),
        CliAction::Agents { args } => println!(
            "{}",
            handle_agents_slash_command(args.as_deref(), &env::current_dir()?)?
        ),
        CliAction::Skills { args } => println!(
            "{}",
            handle_skills_slash_command(args.as_deref(), &env::current_dir()?)?
        ),
        CliAction::Init => run_init()?,
        CliAction::Doctor => run_doctor()?,
        CliAction::SelfUpdate => run_self_update()?,
        CliAction::ResumeSession {
            session_path,
            commands,
        } => resume_session_cli(session_path.as_deref(), &commands)?,
        CliAction::Prompt {
            prompt,
            model,
            allowed_tools,
            permission_mode,
            collaboration_mode,
            reasoning_effort,
            fast_mode,
            output_format,
        } => LiveCli::new(
            model,
            true,
            allowed_tools,
            permission_mode,
            collaboration_mode,
            reasoning_effort,
            fast_mode,
            matches!(output_format, CliOutputFormat::Text),
        )?
        .run_turn_with_output(&prompt, output_format)?,
        CliAction::Repl {
            model,
            allowed_tools,
            permission_mode,
            collaboration_mode,
            reasoning_effort,
            fast_mode,
        } => run_repl(
            model,
            allowed_tools,
            permission_mode,
            collaboration_mode,
            reasoning_effort,
            fast_mode,
        )?,
        CliAction::Login { service, api_key } => login(service, api_key)?,
        CliAction::Logout { service } => logout(service)?,
        CliAction::Help => print_help(),
        CliAction::Version => print_version(),
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
enum CliAction {
    DumpManifests,
    BootstrapPlan,
    PrintSystemPrompt {
        cwd: PathBuf,
        date: String,
    },
    Model {
        model: Option<String>,
    },
    Provider {
        provider: Option<String>,
    },
    Proxy {
        mode: ProxyCommand,
    },
    Mcp {
        action: McpCommand,
    },
    Plugins {
        action: Option<String>,
        target: Option<String>,
    },
    Branch {
        action: Option<String>,
        target: Option<String>,
    },
    Worktree {
        action: Option<String>,
        path: Option<String>,
        branch: Option<String>,
    },
    Agents {
        args: Option<String>,
    },
    Skills {
        args: Option<String>,
    },
    Init,
    Doctor,
    SelfUpdate,
    ResumeSession {
        session_path: Option<PathBuf>,
        commands: Vec<String>,
    },
    Prompt {
        prompt: String,
        model: String,
        allowed_tools: Option<AllowedToolSet>,
        permission_mode: PermissionMode,
        collaboration_mode: CollaborationMode,
        reasoning_effort: Option<ReasoningEffort>,
        fast_mode: FastMode,
        output_format: CliOutputFormat,
    },
    Login {
        service: Option<AuthService>,
        api_key: Option<String>,
    },
    Logout {
        service: Option<AuthService>,
    },
    Repl {
        model: String,
        allowed_tools: Option<AllowedToolSet>,
        permission_mode: PermissionMode,
        collaboration_mode: CollaborationMode,
        reasoning_effort: Option<ReasoningEffort>,
        fast_mode: FastMode,
    },
    Help,
    Version,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CliOutputFormat {
    Text,
    Json,
}

impl CliOutputFormat {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "text" => Ok(Self::Text),
            "json" => Ok(Self::Json),
            other => Err(format!(
                "unsupported value for --output-format: {other} (expected text or json)"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CollaborationMode {
    Build,
    Plan,
}

impl CollaborationMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Build => "build",
            Self::Plan => "plan",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FastMode {
    Off,
    On,
}

impl FastMode {
    const fn enabled(self) -> bool {
        matches!(self, Self::On)
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::On => "on",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AuthService {
    NanoGpt,
    Synthetic,
    OpenAiCodex,
    OpencodeGo,
    Exa,
}

impl AuthService {
    const fn display_name(self) -> &'static str {
        match self {
            Self::NanoGpt => "NanoGPT",
            Self::Synthetic => "Synthetic",
            Self::OpenAiCodex => "OpenAI Codex",
            Self::OpencodeGo => "OpenCode Go",
            Self::Exa => "Exa",
        }
    }

    const fn slug(self) -> &'static str {
        match self {
            Self::NanoGpt => "nanogpt",
            Self::Synthetic => "synthetic",
            Self::OpenAiCodex => "openai-codex",
            Self::OpencodeGo => "opencode-go",
            Self::Exa => "exa",
        }
    }

    const fn credential_key(self) -> &'static str {
        match self {
            Self::NanoGpt => "nanogpt_api_key",
            Self::Synthetic => "synthetic_api_key",
            Self::OpenAiCodex => "openai_codex_auth",
            Self::OpencodeGo => "opencode_go_api_key",
            Self::Exa => "exa_api_key",
        }
    }

    const fn all() -> &'static [AuthService] {
        &[
            Self::NanoGpt,
            Self::Synthetic,
            Self::OpenAiCodex,
            Self::OpencodeGo,
            Self::Exa,
        ]
    }

    const fn runtime_service(self) -> Option<ApiService> {
        match self {
            Self::NanoGpt => Some(ApiService::NanoGpt),
            Self::Synthetic => Some(ApiService::Synthetic),
            Self::OpenAiCodex => Some(ApiService::OpenAiCodex),
            Self::OpencodeGo => Some(ApiService::OpencodeGo),
            Self::Exa => None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAiCodexTokenResponse {
    access_token: String,
    refresh_token: String,
    #[serde(default)]
    id_token: Option<String>,
    #[serde(default)]
    expires_in: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAiCodexDeviceCodeResponse {
    device_auth_id: String,
    user_code: String,
    interval: String,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAiCodexDeviceTokenResponse {
    authorization_code: String,
    code_verifier: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum McpCommand {
    Status,
    Tools,
    Reload,
    Add { name: String },
    Enable { name: String },
    Disable { name: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct McpServerStatus {
    server_name: String,
    scope: ConfigSource,
    enabled: bool,
    transport: McpTransport,
    loaded: bool,
    tool_count: usize,
    note: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct McpToolBinding {
    exposed_name: String,
    server_name: String,
    upstream_name: String,
    description: String,
    input_schema: JsonValue,
    config: ScopedMcpServerConfig,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct McpCatalog {
    servers: Vec<McpServerStatus>,
    tools: Vec<McpToolBinding>,
}

#[derive(Debug, Clone)]
struct SessionHandle {
    id: String,
    path: PathBuf,
}

#[derive(Debug, Clone)]
struct ManagedSessionSummary {
    id: String,
    path: PathBuf,
    modified_epoch_secs: u64,
    message_count: usize,
    model: Option<String>,
    started_at: Option<String>,
    last_prompt: Option<String>,
}

#[derive(Debug, Clone)]
struct SessionRuntimeState {
    model: String,
    service: ApiService,
    allowed_tools: Option<AllowedToolSet>,
    permission_mode: PermissionMode,
    collaboration_mode: CollaborationMode,
    reasoning_effort: Option<ReasoningEffort>,
    fast_mode: FastMode,
    proxy_tool_calls: bool,
}

fn parse_args(args: &[String]) -> Result<CliAction, String> {
    let mut model = resolve_model_alias(&default_model_or(DEFAULT_MODEL)).to_string();
    let mut permission_mode = default_permission_mode();
    let mut collaboration_mode = CollaborationMode::Build;
    let mut reasoning_effort = None;
    let mut fast_mode = FastMode::Off;
    let mut output_format = CliOutputFormat::Text;
    let mut allowed_tool_values = Vec::new();
    let mut rest = Vec::new();
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--model" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "missing value for --model".to_string())?;
                model = resolve_model_alias(value).to_string();
                index += 2;
            }
            flag if flag.starts_with("--model=") => {
                model = resolve_model_alias(&flag[8..]).to_string();
                index += 1;
            }
            "--permission-mode" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "missing value for --permission-mode".to_string())?;
                permission_mode = parse_permission_mode_arg(value)?;
                index += 2;
            }
            flag if flag.starts_with("--permission-mode=") => {
                permission_mode = parse_permission_mode_arg(&flag[18..])?;
                index += 1;
            }
            "--mode" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "missing value for --mode".to_string())?;
                collaboration_mode = parse_collaboration_mode_arg(value)?;
                index += 2;
            }
            flag if flag.starts_with("--mode=") => {
                collaboration_mode = parse_collaboration_mode_arg(&flag[7..])?;
                index += 1;
            }
            "--reasoning" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "missing value for --reasoning".to_string())?;
                reasoning_effort = parse_reasoning_effort_arg(value)?;
                index += 2;
            }
            flag if flag.starts_with("--reasoning=") => {
                reasoning_effort = parse_reasoning_effort_arg(&flag[12..])?;
                index += 1;
            }
            "--thinking" => {
                reasoning_effort = Some(ReasoningEffort::Medium);
                index += 1;
            }
            "--fast" => {
                fast_mode = FastMode::On;
                index += 1;
            }
            "--output-format" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "missing value for --output-format".to_string())?;
                output_format = CliOutputFormat::parse(value)?;
                index += 2;
            }
            flag if flag.starts_with("--output-format=") => {
                output_format = CliOutputFormat::parse(&flag[16..])?;
                index += 1;
            }
            "-p" => {
                let prompt = args[index + 1..].join(" ");
                if prompt.trim().is_empty() {
                    return Err("-p requires a prompt string".to_string());
                }
                return Ok(CliAction::Prompt {
                    prompt,
                    model: resolve_model_alias(&model).to_string(),
                    allowed_tools: normalize_allowed_tools(&allowed_tool_values)?,
                    permission_mode,
                    collaboration_mode,
                    reasoning_effort,
                    fast_mode,
                    output_format,
                });
            }
            "--print" => {
                output_format = CliOutputFormat::Text;
                index += 1;
            }
            "--allowedTools" | "--allowed-tools" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "missing value for --allowedTools".to_string())?;
                allowed_tool_values.push(value.clone());
                index += 2;
            }
            flag if flag.starts_with("--allowedTools=") => {
                allowed_tool_values.push(flag[15..].to_string());
                index += 1;
            }
            flag if flag.starts_with("--allowed-tools=") => {
                allowed_tool_values.push(flag[16..].to_string());
                index += 1;
            }
            other => {
                rest.push(other.to_string());
                index += 1;
            }
        }
    }

    let allowed_tools = normalize_allowed_tools(&allowed_tool_values)?;

    if rest.is_empty() {
        return Ok(CliAction::Repl {
            model,
            allowed_tools,
            permission_mode,
            collaboration_mode,
            reasoning_effort,
            fast_mode,
        });
    }
    if matches!(rest.first().map(String::as_str), Some("--help" | "-h")) {
        return Ok(CliAction::Help);
    }
    if matches!(rest.first().map(String::as_str), Some("--version" | "-V")) {
        return Ok(CliAction::Version);
    }
    if rest.first().map(String::as_str) == Some("--resume") {
        return parse_resume_args(&rest[1..]);
    }

    match rest[0].as_str() {
        "dump-manifests" => Ok(CliAction::DumpManifests),
        "bootstrap-plan" => Ok(CliAction::BootstrapPlan),
        "system-prompt" => parse_system_prompt_args(&rest[1..]),
        "login" | "auth" => parse_login_args(&rest[1..]),
        "logout" => parse_logout_args(&rest[1..]),
        "model" | "models" => parse_model_args(&rest[1..]),
        "provider" | "providers" => parse_provider_args(&rest[1..]),
        "proxy" => parse_proxy_args(&rest[1..]),
        "mcp" => parse_mcp_args(&rest[1..]),
        "resume" => parse_resume_args(&rest[1..]),
        "plugins" | "plugin" | "marketplace" => Ok(CliAction::Plugins {
            action: rest.get(1).cloned(),
            target: {
                let remainder = rest.iter().skip(2).cloned().collect::<Vec<_>>().join(" ");
                (!remainder.is_empty()).then_some(remainder)
            },
        }),
        "branch" => Ok(CliAction::Branch {
            action: rest.get(1).cloned(),
            target: rest.get(2).cloned(),
        }),
        "worktree" => Ok(CliAction::Worktree {
            action: rest.get(1).cloned(),
            path: rest.get(2).cloned(),
            branch: rest.get(3).cloned(),
        }),
        "agents" => Ok(CliAction::Agents {
            args: join_optional_args(&rest[1..]),
        }),
        "skills" => Ok(CliAction::Skills {
            args: join_optional_args(&rest[1..]),
        }),
        "init" => Ok(CliAction::Init),
        "doctor" => Ok(CliAction::Doctor),
        "self-update" => Ok(CliAction::SelfUpdate),
        "prompt" => {
            let prompt = rest[1..].join(" ");
            if prompt.trim().is_empty() {
                return Err("prompt subcommand requires a prompt string".to_string());
            }
            Ok(CliAction::Prompt {
                prompt,
                model,
                allowed_tools,
                permission_mode,
                collaboration_mode,
                reasoning_effort,
                fast_mode,
                output_format,
            })
        }
        other if other.starts_with('/') => parse_direct_slash_cli_action(&rest),
        other if !other.starts_with('/') => Ok(CliAction::Prompt {
            prompt: rest.join(" "),
            model,
            allowed_tools,
            permission_mode,
            collaboration_mode,
            reasoning_effort,
            fast_mode,
            output_format,
        }),
        other => Err(format!("unknown subcommand: {other}")),
    }
}

fn resolve_model_alias(model: &str) -> &str {
    match model.trim().to_ascii_lowercase().as_str() {
        "default" | "glm" | "glm5.1" | "glm-5.1" | "glm_5_1" | "zai-org/glm-5.1" => {
            "zai-org/glm-5.1"
        }
        "glm5" | "glm-5" | "glm_5" | "zai-org/glm-5" => "zai-org/glm-5",
        _ => model,
    }
}

fn parse_direct_slash_cli_action(rest: &[String]) -> Result<CliAction, String> {
    let raw = rest.join(" ");
    match SlashCommand::parse(&raw) {
        Some(SlashCommand::Help { .. }) => Ok(CliAction::Help),
        Some(SlashCommand::Agents { args }) => Ok(CliAction::Agents { args }),
        Some(SlashCommand::Skills { args }) => Ok(CliAction::Skills { args }),
        Some(command) => Err(format!(
            "unsupported direct slash command outside the REPL: {command_name}",
            command_name = match command {
                SlashCommand::Unknown(name) => format!("/{name}"),
                _ => rest[0].clone(),
            }
        )),
        None => Err(format!("unknown subcommand: {}", rest[0])),
    }
}

fn join_optional_args(args: &[String]) -> Option<String> {
    let joined = args.join(" ");
    let trimmed = joined.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn normalize_allowed_tools(values: &[String]) -> Result<Option<AllowedToolSet>, String> {
    current_tool_registry()?.normalize_allowed_tools(values)
}

fn parse_system_prompt_args(args: &[String]) -> Result<CliAction, String> {
    let mut cwd = env::current_dir().map_err(|error| error.to_string())?;
    let mut date = DEFAULT_DATE.to_string();
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--cwd" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "missing value for --cwd".to_string())?;
                cwd = PathBuf::from(value);
                index += 2;
            }
            "--date" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "missing value for --date".to_string())?;
                date.clone_from(value);
                index += 2;
            }
            other => return Err(format!("unknown system-prompt option: {other}")),
        }
    }

    Ok(CliAction::PrintSystemPrompt { cwd, date })
}

fn parse_model_args(args: &[String]) -> Result<CliAction, String> {
    if args.len() > 1 {
        return Err("model accepts at most one optional model id".to_string());
    }
    Ok(CliAction::Model {
        model: args.first().cloned(),
    })
}

fn parse_provider_args(args: &[String]) -> Result<CliAction, String> {
    if args.len() > 1 {
        return Err("provider accepts at most one optional provider id".to_string());
    }
    Ok(CliAction::Provider {
        provider: args.first().cloned(),
    })
}

fn parse_login_args(args: &[String]) -> Result<CliAction, String> {
    let parsed = parse_login_tokens(args.iter().map(String::as_str).collect())?;
    Ok(CliAction::Login {
        service: parsed.service,
        api_key: parsed.api_key,
    })
}

fn parse_logout_args(args: &[String]) -> Result<CliAction, String> {
    let parsed = parse_logout_tokens(args.iter().map(String::as_str).collect())?;
    Ok(CliAction::Logout {
        service: parsed.service,
    })
}

fn parse_login_tokens(tokens: Vec<&str>) -> Result<LoginCommand, String> {
    let mut service = None;
    let mut api_key = None;
    let mut index = 0;

    while index < tokens.len() {
        match tokens[index] {
            "--service" => {
                let value = tokens
                    .get(index + 1)
                    .ok_or_else(|| "missing value for --service".to_string())?;
                service = Some(parse_login_service(value)?);
                index += 2;
            }
            flag if flag.starts_with("--service=") => {
                service = Some(parse_login_service(&flag[10..])?);
                index += 1;
            }
            "--api-key" => {
                let value = tokens
                    .get(index + 1)
                    .ok_or_else(|| "missing value for --api-key".to_string())?;
                api_key = Some((*value).to_string());
                index += 2;
            }
            flag if flag.starts_with("--api-key=") => {
                api_key = Some(flag[10..].to_string());
                index += 1;
            }
            value
                if matches!(
                    value,
                    "nanogpt"
                        | "nano-gpt"
                        | "nano"
                        | "synthetic"
                        | "synthetic.new"
                        | "openai-codex"
                        | "openai_codex"
                        | "chatgpt"
                        | "opencode-go"
                        | "opencodego"
                        | "exa"
                ) && api_key.is_none() =>
            {
                service = Some(parse_login_service(value)?);
                index += 1;
            }
            value if api_key.is_none() => {
                api_key = Some(value.to_string());
                index += 1;
            }
            other => return Err(format!("unexpected login argument: {other}")),
        }
    }

    Ok(LoginCommand { service, api_key })
}

fn parse_login_service(value: &str) -> Result<AuthService, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "nanogpt" | "nano-gpt" | "nano" => Ok(AuthService::NanoGpt),
        "synthetic" | "synthetic.new" => Ok(AuthService::Synthetic),
        "openai-codex" | "openai_codex" | "chatgpt" => Ok(AuthService::OpenAiCodex),
        "opencode-go" | "opencodego" => Ok(AuthService::OpencodeGo),
        "exa" => Ok(AuthService::Exa),
        other => Err(format!(
            "unsupported login service `{other}`; expected nanogpt, synthetic, openai-codex, opencode-go, or exa"
        )),
    }
}

fn parse_logout_tokens(tokens: Vec<&str>) -> Result<LogoutCommand, String> {
    let mut service = None;
    let mut index = 0;

    while index < tokens.len() {
        match tokens[index] {
            "--service" => {
                let value = tokens
                    .get(index + 1)
                    .ok_or_else(|| "missing value for --service".to_string())?;
                service = Some(parse_login_service(value)?);
                index += 2;
            }
            flag if flag.starts_with("--service=") => {
                service = Some(parse_login_service(&flag[10..])?);
                index += 1;
            }
            value
                if matches!(
                    value,
                    "nanogpt"
                        | "nano-gpt"
                        | "nano"
                        | "synthetic"
                        | "synthetic.new"
                        | "openai-codex"
                        | "openai_codex"
                        | "chatgpt"
                        | "opencode-go"
                        | "opencodego"
                        | "exa"
                ) =>
            {
                service = Some(parse_login_service(value)?);
                index += 1;
            }
            other => return Err(format!("unexpected logout argument: {other}")),
        }
    }

    Ok(LogoutCommand { service })
}

fn parse_proxy_args(args: &[String]) -> Result<CliAction, String> {
    if args.len() > 1 {
        return Err("proxy accepts at most one optional argument".to_string());
    }
    Ok(CliAction::Proxy {
        mode: parse_proxy_value(args.first().map(String::as_str))?,
    })
}

fn parse_mcp_args(args: &[String]) -> Result<CliAction, String> {
    let action = match args.first().map(String::as_str) {
        None | Some("status") => McpCommand::Status,
        Some("tools") => McpCommand::Tools,
        Some("reload") => McpCommand::Reload,
        Some("add") => {
            let name = args
                .get(1)
                .ok_or_else(|| "mcp add requires a server name".to_string())?;
            if args.len() > 2 {
                return Err("mcp add accepts exactly one server name".to_string());
            }
            McpCommand::Add { name: name.clone() }
        }
        Some("enable") => {
            let name = args
                .get(1)
                .ok_or_else(|| "mcp enable requires a server name".to_string())?;
            if args.len() > 2 {
                return Err("mcp enable accepts exactly one server name".to_string());
            }
            McpCommand::Enable { name: name.clone() }
        }
        Some("disable") => {
            let name = args
                .get(1)
                .ok_or_else(|| "mcp disable requires a server name".to_string())?;
            if args.len() > 2 {
                return Err("mcp disable accepts exactly one server name".to_string());
            }
            McpCommand::Disable { name: name.clone() }
        }
        Some(other) => {
            return Err(format!(
                "mcp accepts status, tools, reload, add <name>, enable <name>, or disable <name> (got {other})"
            ));
        }
    };
    if !matches!(
        action,
        McpCommand::Add { .. } | McpCommand::Enable { .. } | McpCommand::Disable { .. }
    ) && args.len() > 1
    {
        return Err(
            "mcp accepts at most one optional argument unless using add <name>, enable <name>, or disable <name>".to_string(),
        );
    }
    Ok(CliAction::Mcp { action })
}

fn parse_resume_args(args: &[String]) -> Result<CliAction, String> {
    let (session_path, commands) = match args.first() {
        None => (None, Vec::new()),
        Some(first) if first.trim_start().starts_with('/') => {
            return Err("resume without a session id/path opens the session picker and does not accept trailing commands".to_string());
        }
        Some(first) => (Some(PathBuf::from(first)), args[1..].to_vec()),
    };
    if commands
        .iter()
        .any(|command| !command.trim_start().starts_with('/'))
    {
        return Err("--resume trailing arguments must be slash commands".to_string());
    }
    Ok(CliAction::ResumeSession {
        session_path,
        commands,
    })
}

fn dump_manifests() {
    let workspace_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let paths = UpstreamPaths::from_workspace_dir(&workspace_dir);
    match extract_manifest(&paths) {
        Ok(manifest) => {
            println!("commands: {}", manifest.commands.entries().len());
            println!("tools: {}", manifest.tools.entries().len());
            println!("bootstrap phases: {}", manifest.bootstrap.phases().len());
        }
        Err(error) => {
            eprintln!("failed to extract manifests: {error}");
            std::process::exit(1);
        }
    }
}

fn print_bootstrap_plan() {
    for phase in runtime::BootstrapPlan::pebble_default().phases() {
        println!("- {phase:?}");
    }
}

fn print_system_prompt(cwd: PathBuf, date: String) {
    let model = default_model_or(DEFAULT_MODEL);
    let service = infer_service_for_model(&model);
    match load_system_prompt_with_model_family(
        cwd,
        date,
        env::consts::OS,
        "unknown",
        prompt_model_family(service, &model),
    ) {
        Ok(sections) => println!("{}", sections.join("\n\n")),
        Err(error) => {
            eprintln!("failed to build system prompt: {error}");
            std::process::exit(1);
        }
    }
}

fn resume_session_cli(
    session_path: Option<&Path>,
    commands: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let handle = match session_path {
        Some(session_path) => resolve_session_reference(&session_path.display().to_string())?,
        None => match prompt_for_session_selection(None)? {
            Some(handle) => handle,
            None => return Ok(()),
        },
    };
    let session = match Session::load_from_path(&handle.path) {
        Ok(session) => session,
        Err(error) => {
            eprintln!("failed to restore session: {error}");
            std::process::exit(1);
        }
    };

    if commands.is_empty() {
        return run_repl_from_session(handle, session);
    }

    let mut session = session;
    for raw_command in commands {
        let Some(command) = SlashCommand::parse(raw_command) else {
            eprintln!("unsupported resumed command: {raw_command}");
            std::process::exit(2);
        };
        match run_resume_command(&handle.path, &session, &command) {
            Ok(ResumeCommandOutcome {
                session: next_session,
                message,
            }) => {
                session = next_session.clone();
                if let Err(error) = next_session.save_to_path(&handle.path) {
                    eprintln!("failed to persist resumed session: {error}");
                    std::process::exit(1);
                }
                if let Some(message) = message {
                    println!("{message}");
                }
            }
            Err(error) => {
                eprintln!("{error}");
                std::process::exit(2);
            }
        }
    }
    Ok(())
}

fn prompt_for_session_selection(
    active_session_id: Option<&str>,
) -> Result<Option<SessionHandle>, Box<dyn std::error::Error>> {
    let sessions = list_managed_sessions()?;
    if sessions.is_empty() {
        println!("No managed sessions saved yet.");
        return Ok(None);
    }

    print!("Filter sessions (optional, press Enter for all): ");
    io::stdout().flush()?;
    let mut filter = String::new();
    io::stdin().read_line(&mut filter)?;
    let filter = filter.trim().to_ascii_lowercase();
    let sessions = if filter.is_empty() {
        sessions
    } else {
        sessions
            .into_iter()
            .filter(|session| {
                fuzzy_session_match(&session.id, &filter)
                    || session
                        .model
                        .as_deref()
                        .is_some_and(|model| fuzzy_session_match(model, &filter))
                    || session
                        .last_prompt
                        .as_deref()
                        .is_some_and(|prompt| fuzzy_session_match(prompt, &filter))
            })
            .collect::<Vec<_>>()
    };
    if sessions.is_empty() {
        println!("No sessions matched that filter.");
        return Ok(None);
    }

    println!("Recent sessions");
    for (index, session) in sessions.iter().enumerate() {
        let marker = if active_session_id == Some(session.id.as_str()) {
            "current"
        } else if index == 0 {
            "last"
        } else {
            "saved"
        };
        let model = session.model.as_deref().unwrap_or("unknown");
        let last_prompt = session.last_prompt.as_deref().map_or_else(
            || "-".to_string(),
            |prompt| truncate_for_summary(prompt, 48),
        );
        println!(
            "  {idx:>2}. {id:<22} {marker:<7} model={model:<24} msgs={msgs:<4} last={last}",
            idx = index + 1,
            id = session.id,
            marker = marker,
            model = model,
            msgs = session.message_count,
            last = last_prompt,
        );
    }
    println!();
    print!(
        "Select a session to resume [1-{}] or press Enter to cancel: ",
        sessions.len()
    );
    io::stdout().flush()?;
    let mut buffer = String::new();
    io::stdin().read_line(&mut buffer)?;
    let selection = buffer.trim();
    if selection.is_empty() {
        return Ok(None);
    }
    let index = selection
        .parse::<usize>()
        .map_err(|_| format!("invalid selection: {selection}"))?;
    let Some(session) = sessions.get(index.saturating_sub(1)) else {
        return Err(format!("selection out of range: {selection}").into());
    };
    Ok(Some(SessionHandle {
        id: session.id.clone(),
        path: session.path.clone(),
    }))
}

fn fuzzy_session_match(haystack: &str, query: &str) -> bool {
    let haystack = haystack.to_ascii_lowercase();
    if haystack.contains(query) {
        return true;
    }

    let mut query_chars = query.chars();
    let mut current = match query_chars.next() {
        Some(ch) => ch,
        None => return true,
    };

    for hay in haystack.chars() {
        if hay == current {
            match query_chars.next() {
                Some(next) => current = next,
                None => return true,
            }
        }
    }
    false
}

fn prompt_for_auth_service_selection() -> Result<Option<AuthService>, Box<dyn std::error::Error>> {
    println!("Auth services");
    for (index, service) in AuthService::all().iter().enumerate() {
        println!(
            "  {:>2}. {} ({})",
            index + 1,
            service.display_name(),
            service.slug()
        );
    }
    println!();
    print!(
        "Select a service [1-{}] or press Enter to cancel: ",
        AuthService::all().len()
    );
    io::stdout().flush()?;
    let mut buffer = String::new();
    io::stdin().read_line(&mut buffer)?;
    let selection = buffer.trim();
    if selection.is_empty() {
        return Ok(None);
    }
    let index = selection
        .parse::<usize>()
        .map_err(|_| format!("invalid selection: {selection}"))?;
    let Some(service) = AuthService::all().get(index.saturating_sub(1)) else {
        return Err(format!("selection out of range: {selection}").into());
    };
    Ok(Some(*service))
}

fn login(
    service: Option<AuthService>,
    api_key: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let service = match service {
        Some(service) => service,
        None => match prompt_for_auth_service_selection()? {
            Some(service) => service,
            None => return Ok(()),
        },
    };
    if service == AuthService::OpenAiCodex {
        if api_key.is_some() {
            return Err(
                "OpenAI Codex login uses device-code authentication and does not accept API keys"
                    .into(),
            );
        }
        let credentials_path = login_openai_codex()?;
        println!(
            "Saved {} credentials to {}",
            service.display_name(),
            credentials_path.display()
        );
        if let Some(note) = login_model_guidance(service) {
            println!("{note}");
        }
        return Ok(());
    }
    let api_key = resolve_auth_api_key(service, api_key)?;
    let credentials_path = save_credentials(service, &api_key)?;
    println!(
        "Saved {} credentials to {}",
        service.display_name(),
        credentials_path.display()
    );
    if let Some(note) = login_model_guidance(service) {
        println!("{note}");
    }
    Ok(())
}

fn login_model_guidance(service: AuthService) -> Option<String> {
    let target_service = service.runtime_service()?;
    let active_model = default_model_or(DEFAULT_MODEL);
    let active_service = infer_service_for_model(&active_model);
    if active_service == target_service {
        return None;
    }

    let switch_hint = match service {
        AuthService::Synthetic => {
            "Run `/model` and choose a Synthetic model id (usually prefixed with `hf:`)."
        }
        AuthService::OpenAiCodex => {
            "Run `/model` and choose an OpenAI Codex model id prefixed with `openai-codex/`."
        }
        AuthService::OpencodeGo => {
            "Run `/model` and choose an OpenCode Go model id prefixed with `opencode-go/`."
        }
        AuthService::NanoGpt => {
            "Run `/model zai-org/glm-5.1` or another NanoGPT-backed model if you want to use this key immediately."
        }
        AuthService::Exa => return None,
    };

    Some(format!(
        "Note: your current model is `{active_model}` on {}. Logging into {} saves credentials but does not switch the active model. {switch_hint}",
        active_service.display_name(),
        service.display_name(),
    ))
}

fn logout(service: Option<AuthService>) -> Result<(), Box<dyn std::error::Error>> {
    let service = match service {
        Some(service) => service,
        None => match prompt_for_auth_service_selection()? {
            Some(service) => service,
            None => return Ok(()),
        },
    };

    let outcome = remove_saved_credentials(service)?;
    match outcome {
        CredentialRemovalOutcome::Removed { path } => {
            println!(
                "Removed saved {} credentials from {}",
                service.display_name(),
                path.display()
            );
        }
        CredentialRemovalOutcome::Missing { path } => {
            println!(
                "No saved {} credentials found in {}",
                service.display_name(),
                path.display()
            );
        }
    }
    Ok(())
}

fn login_openai_codex() -> Result<PathBuf, Box<dyn std::error::Error>> {
    login_openai_codex_device_code()
}

fn login_openai_codex_device_code() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let client = BlockingClient::builder()
        .timeout(Duration::from_secs(20))
        .build()?;
    let response = client
        .post(format!(
            "{OPENAI_CODEX_ISSUER}/api/accounts/deviceauth/usercode"
        ))
        .header("content-type", "application/json")
        .json(&serde_json::json!({ "client_id": OPENAI_CODEX_CLIENT_ID }))
        .send()?;
    let response = response.error_for_status()?;
    let payload: OpenAiCodexDeviceCodeResponse = response.json()?;
    let interval_secs = payload
        .interval
        .trim()
        .parse::<u64>()
        .ok()
        .filter(|value| *value > 0)
        .unwrap_or(5);

    println!(
        "Open this URL in your browser and sign in with ChatGPT:\n\n{issuer}/codex/device\n\nEnter this one-time code:\n\n{code}\n",
        issuer = OPENAI_CODEX_ISSUER,
        code = payload.user_code
    );

    let deadline = std::time::Instant::now() + OPENAI_CODEX_DEVICE_AUTH_TIMEOUT;
    loop {
        let response = client
            .post(format!(
                "{OPENAI_CODEX_ISSUER}/api/accounts/deviceauth/token"
            ))
            .header("content-type", "application/json")
            .json(&serde_json::json!({
                "device_auth_id": payload.device_auth_id,
                "user_code": payload.user_code,
            }))
            .send()?;

        if response.status().is_success() {
            let payload: OpenAiCodexDeviceTokenResponse = response.json()?;
            let tokens = exchange_openai_codex_authorization_code(
                &payload.authorization_code,
                &format!("{OPENAI_CODEX_ISSUER}/deviceauth/callback"),
                &payload.code_verifier,
            )?;
            return persist_openai_codex_tokens(tokens);
        }

        let status = response.status();
        if !matches!(status.as_u16(), 403 | 404) {
            let body = response.text().unwrap_or_default();
            return Err(format!(
                "device code authorization failed with status {}: {}",
                status, body
            )
            .into());
        }

        if std::time::Instant::now() >= deadline {
            return Err("device code authorization timed out after 15 minutes".into());
        }

        std::thread::sleep(
            Duration::from_secs(interval_secs)
                .saturating_add(OPENAI_CODEX_DEVICE_POLL_SAFETY_MARGIN),
        );
    }
}

fn exchange_openai_codex_authorization_code(
    code: &str,
    redirect_uri: &str,
    verifier: &str,
) -> Result<OpenAiCodexTokenResponse, Box<dyn std::error::Error>> {
    let client = BlockingClient::builder()
        .timeout(Duration::from_secs(20))
        .build()?;
    let response = client
        .post(format!("{OPENAI_CODEX_ISSUER}/oauth/token"))
        .header("content-type", "application/x-www-form-urlencoded")
        .form(&[
            ("grant_type", "authorization_code"),
            ("code", code),
            ("redirect_uri", redirect_uri),
            ("client_id", OPENAI_CODEX_CLIENT_ID),
            ("code_verifier", verifier),
        ])
        .send()?;
    let response = response.error_for_status()?;
    Ok(response.json()?)
}

fn persist_openai_codex_tokens(
    tokens: OpenAiCodexTokenResponse,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let expires_at = tokens
        .expires_in
        .map(|seconds| current_epoch_millis().saturating_add(seconds.saturating_mul(1_000)));
    let account_id = tokens
        .id_token
        .as_deref()
        .and_then(extract_openai_codex_account_id)
        .or_else(|| extract_openai_codex_account_id(&tokens.access_token));
    let credentials = OpenAiCodexCredentials {
        access_token: tokens.access_token,
        refresh_token: tokens.refresh_token,
        expires_at,
        account_id,
    };
    Ok(save_openai_codex_credentials(&credentials)?)
}

fn extract_openai_codex_account_id(token: &str) -> Option<String> {
    let claims = decode_jwt_claims(token)?;
    claims
        .get("chatgpt_account_id")
        .and_then(JsonValue::as_str)
        .map(ToOwned::to_owned)
        .or_else(|| {
            claims
                .get("https://api.openai.com/auth")
                .and_then(JsonValue::as_object)
                .and_then(|value| value.get("chatgpt_account_id"))
                .and_then(JsonValue::as_str)
                .map(ToOwned::to_owned)
        })
        .or_else(|| {
            claims
                .get("organizations")
                .and_then(JsonValue::as_array)
                .and_then(|values| values.first())
                .and_then(JsonValue::as_object)
                .and_then(|value| value.get("id"))
                .and_then(JsonValue::as_str)
                .map(ToOwned::to_owned)
        })
}

fn decode_jwt_claims(token: &str) -> Option<JsonValue> {
    let payload = token.split('.').nth(1)?;
    let decoded = decode_base64_url(payload)?;
    serde_json::from_slice(&decoded).ok()
}

fn decode_base64_url(value: &str) -> Option<Vec<u8>> {
    let mut output = Vec::new();
    let mut buffer = 0_u32;
    let mut bits = 0_u8;

    for byte in value.bytes() {
        let sextet = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'-' => 62,
            b'_' => 63,
            b'=' => break,
            _ => return None,
        };
        buffer = (buffer << 6) | u32::from(sextet);
        bits += 6;
        while bits >= 8 {
            bits -= 8;
            output.push(((buffer >> bits) & 0xFF) as u8);
        }
    }

    Some(output)
}

fn handle_model_action(model: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    match model {
        Some(model) => {
            let model = resolve_model_alias(&model).to_string();
            persist_current_model(model.clone())?;
            println!("Selected model: {model}");
        }
        None => match open_model_picker()?.selected_model {
            Some(model) => println!("Selected model: {model}"),
            None => println!("Model selection cancelled."),
        },
    }
    Ok(())
}

fn handle_provider_action(provider: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    let model = default_model_or(DEFAULT_MODEL);
    if infer_service_for_model(&model) != ApiService::NanoGpt {
        return Err(format!(
            "provider overrides are only available for NanoGPT models; current model {} is on {}",
            model,
            infer_service_for_model(&model).display_name()
        )
        .into());
    }
    match provider {
        Some(provider) if is_clear_provider_arg(&provider) => {
            persist_provider_for_model(&model, None)?;
            println!("Cleared provider override for current model: {model}");
        }
        Some(provider) => {
            validate_provider_for_model(&model, &provider)?;
            persist_provider_for_model(&model, Some(provider.clone()))?;
            println!(
                "Selected provider override for current model {model}: {provider} (paygo routing enabled)"
            );
        }
        None => match open_provider_picker(&model)?.selected_provider {
            Some(provider) => {
                println!(
                    "Selected provider override for current model {model}: {provider} (paygo routing enabled)"
                )
            }
            None => println!("Using platform default provider for current model {model}."),
        },
    }
    Ok(())
}

fn provider_label_for_service_model(service: ApiService, model: &str) -> Option<String> {
    (service == ApiService::NanoGpt)
        .then(|| provider_for_model(model).unwrap_or_else(|| "<platform default>".to_string()))
}

fn handle_proxy_action(mode: ProxyCommand) -> Result<(), Box<dyn std::error::Error>> {
    let current = proxy_tool_calls_enabled();
    let next = match mode {
        ProxyCommand::Toggle => !current,
        ProxyCommand::Enable => true,
        ProxyCommand::Disable => false,
        ProxyCommand::Status => current,
    };
    if !matches!(mode, ProxyCommand::Status) {
        persist_proxy_tool_calls(next)?;
    }
    println!(
        "Proxy tool-call translation: {}",
        if next { "enabled" } else { "disabled" }
    );
    if next {
        println!(
            "Native tool schemas are disabled; tool use now goes through XML <tool_call> blocks."
        );
    }
    Ok(())
}

fn handle_mcp_action(action: McpCommand) -> Result<(), Box<dyn std::error::Error>> {
    let cwd = env::current_dir()?;
    match action {
        McpCommand::Status | McpCommand::Reload => {
            let catalog = load_mcp_catalog(&cwd)?;
            if matches!(action, McpCommand::Reload) {
                println!("Reloaded MCP config from {}", cwd.display());
            }
            print_mcp_status(&catalog);
        }
        McpCommand::Tools => {
            let catalog = load_mcp_catalog(&cwd)?;
            print_mcp_tools(&catalog);
        }
        McpCommand::Add { name } => println!("{}", add_mcp_server_interactive(&cwd, &name)?),
        McpCommand::Enable { name } => println!("{}", set_mcp_server_enabled(&cwd, &name, true)?),
        McpCommand::Disable { name } => {
            println!("{}", set_mcp_server_enabled(&cwd, &name, false)?)
        }
    }
    Ok(())
}

fn add_mcp_server_interactive(
    cwd: &Path,
    name: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    if name.trim().is_empty() {
        return Err("mcp server name cannot be empty".into());
    }
    let normalized_name = name.trim();
    println!("Add MCP server: {normalized_name}");
    print!("Transport [stdio/http] (default: stdio): ");
    io::stdout().flush()?;
    let mut transport = String::new();
    io::stdin().read_line(&mut transport)?;
    let transport = match transport.trim().to_ascii_lowercase().as_str() {
        "" | "stdio" => "stdio",
        "http" => "http",
        other => return Err(format!("unsupported transport: {other}").into()),
    };

    let server_config = if transport == "stdio" {
        let command = prompt_text("Command: ")?;
        if command.trim().is_empty() {
            return Err("stdio MCP command cannot be empty".into());
        }
        let args = prompt_text("Args (space-separated, optional): ")?;
        let args = args
            .split_whitespace()
            .map(|value| JsonValue::String(value.to_string()))
            .collect::<Vec<_>>();
        serde_json::json!({
            "type": "stdio",
            "command": command.trim(),
            "args": args,
        })
    } else {
        let url = prompt_text("URL: ")?;
        if url.trim().is_empty() {
            return Err("http MCP url cannot be empty".into());
        }
        serde_json::json!({
            "type": "http",
            "url": url.trim(),
        })
    };

    let settings_dir = cwd.join(".pebble");
    fs::create_dir_all(&settings_dir)?;
    let settings_path = settings_dir.join("settings.json");
    let mut root = match fs::read_to_string(&settings_path) {
        Ok(contents) => serde_json::from_str::<serde_json::Value>(&contents)
            .unwrap_or_else(|_| serde_json::json!({})),
        Err(error) if error.kind() == io::ErrorKind::NotFound => serde_json::json!({}),
        Err(error) => return Err(Box::new(error)),
    };
    if !root.is_object() {
        root = serde_json::json!({});
    }
    let Some(root_object) = root.as_object_mut() else {
        return Err("settings root must be an object".into());
    };
    let mcp_servers = root_object
        .entry("mcpServers")
        .or_insert_with(|| serde_json::json!({}));
    if !mcp_servers.is_object() {
        *mcp_servers = serde_json::json!({});
    }
    let Some(servers_object) = mcp_servers.as_object_mut() else {
        return Err("mcpServers must be an object".into());
    };
    servers_object.insert(normalized_name.to_string(), server_config);
    fs::write(&settings_path, serde_json::to_string_pretty(&root)?)?;

    Ok(format!(
        "MCP\n  result:  added\n  server:  {normalized_name}\n  file:    {}\n  next:    run /mcp reload",
        settings_path.display()
    ))
}

fn set_mcp_server_enabled(
    cwd: &Path,
    name: &str,
    enabled: bool,
) -> Result<String, Box<dyn std::error::Error>> {
    let normalized_name = name.trim();
    if normalized_name.is_empty() {
        return Err("mcp server name cannot be empty".into());
    }

    let config = ConfigLoader::default_for(cwd).load()?;
    let scoped = config
        .mcp()
        .get(normalized_name)
        .ok_or_else(|| format!("unknown MCP server: {normalized_name}"))?;

    let settings_dir = cwd.join(".pebble");
    fs::create_dir_all(&settings_dir)?;
    let settings_path = settings_dir.join("settings.local.json");
    let mut root = match fs::read_to_string(&settings_path) {
        Ok(contents) => serde_json::from_str::<serde_json::Value>(&contents)
            .unwrap_or_else(|_| serde_json::json!({})),
        Err(error) if error.kind() == io::ErrorKind::NotFound => serde_json::json!({}),
        Err(error) => return Err(Box::new(error)),
    };
    if !root.is_object() {
        root = serde_json::json!({});
    }
    let Some(root_object) = root.as_object_mut() else {
        return Err("settings root must be an object".into());
    };
    let mcp_servers = root_object
        .entry("mcpServers")
        .or_insert_with(|| serde_json::json!({}));
    if !mcp_servers.is_object() {
        *mcp_servers = serde_json::json!({});
    }
    let Some(servers_object) = mcp_servers.as_object_mut() else {
        return Err("mcpServers must be an object".into());
    };
    servers_object.insert(
        normalized_name.to_string(),
        mcp_server_config_to_json(&scoped.config, enabled),
    );
    fs::write(&settings_path, serde_json::to_string_pretty(&root)?)?;

    Ok(format!(
        "MCP\n  result:  {}\n  server:  {normalized_name}\n  file:    {}\n  next:    run /mcp reload",
        if enabled { "enabled" } else { "disabled" },
        settings_path.display()
    ))
}

fn mcp_server_config_to_json(
    config: &runtime::McpServerConfig,
    enabled: bool,
) -> serde_json::Value {
    match config {
        runtime::McpServerConfig::Stdio(config) => serde_json::json!({
            "type": "stdio",
            "command": config.command,
            "args": config.args,
            "env": config.env,
            "stderr": match config.stderr {
                runtime::McpStdioStderrMode::Inherit => "inherit",
                runtime::McpStdioStderrMode::Null => "null",
            },
            "enabled": enabled,
        }),
        runtime::McpServerConfig::Sse(config) => serde_json::json!({
            "type": "sse",
            "url": config.url,
            "headers": config.headers,
            "headersHelper": config.headers_helper,
            "oauth": mcp_oauth_to_json(config.oauth.as_ref()),
            "enabled": enabled,
        }),
        runtime::McpServerConfig::Http(config) => serde_json::json!({
            "type": "http",
            "url": config.url,
            "headers": config.headers,
            "headersHelper": config.headers_helper,
            "oauth": mcp_oauth_to_json(config.oauth.as_ref()),
            "enabled": enabled,
        }),
        runtime::McpServerConfig::Ws(config) => serde_json::json!({
            "type": "ws",
            "url": config.url,
            "headers": config.headers,
            "headersHelper": config.headers_helper,
            "enabled": enabled,
        }),
        runtime::McpServerConfig::Sdk(config) => serde_json::json!({
            "type": "sdk",
            "name": config.name,
            "enabled": enabled,
        }),
        runtime::McpServerConfig::ClaudeAiProxy(config) => serde_json::json!({
            "type": "claudeai-proxy",
            "url": config.url,
            "id": config.id,
            "enabled": enabled,
        }),
    }
}

fn mcp_oauth_to_json(oauth: Option<&runtime::McpOAuthConfig>) -> serde_json::Value {
    oauth.map_or(serde_json::Value::Null, |oauth| {
        serde_json::json!({
            "clientId": oauth.client_id,
            "callbackPort": oauth.callback_port,
            "authServerMetadataUrl": oauth.auth_server_metadata_url,
            "xaa": oauth.xaa,
        })
    })
}

fn prompt_text(prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
    print!("{prompt}");
    io::stdout().flush()?;
    let mut buffer = String::new();
    io::stdin().read_line(&mut buffer)?;
    Ok(buffer.trim().to_string())
}

fn resolve_auth_api_key(
    service: AuthService,
    api_key: Option<String>,
) -> Result<String, Box<dyn std::error::Error>> {
    if service == AuthService::OpenAiCodex {
        return Err(
            "OpenAI Codex login uses device-code authentication and does not accept API keys"
                .into(),
        );
    }
    match api_key {
        Some(api_key) if !api_key.trim().is_empty() => Ok(api_key),
        Some(_) => Err(format!("{} API key cannot be empty", service.display_name()).into()),
        None => {
            let api_key = read_secret(&format!("{} API key: ", service.display_name()))?;
            if api_key.trim().is_empty() {
                return Err(format!("{} API key cannot be empty", service.display_name()).into());
            }
            Ok(api_key)
        }
    }
}

fn save_credentials(
    service: AuthService,
    api_key: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let config_home = pebble_config_home()?;
    fs::create_dir_all(&config_home)?;
    let credentials_path = config_home.join("credentials.json");
    let mut parsed = match fs::read_to_string(&credentials_path) {
        Ok(contents) => serde_json::from_str::<serde_json::Value>(&contents)
            .unwrap_or_else(|_| serde_json::json!({})),
        Err(error) if error.kind() == io::ErrorKind::NotFound => serde_json::json!({}),
        Err(error) => return Err(Box::new(error)),
    };
    if !parsed.is_object() {
        parsed = serde_json::json!({});
    }
    let key_name = service.credential_key();
    parsed[key_name] = serde_json::Value::String(api_key.to_string());
    fs::write(&credentials_path, serde_json::to_string_pretty(&parsed)?)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&credentials_path, fs::Permissions::from_mode(0o600))?;
    }
    Ok(credentials_path)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CredentialRemovalOutcome {
    Removed { path: PathBuf },
    Missing { path: PathBuf },
}

fn remove_saved_credentials(
    service: AuthService,
) -> Result<CredentialRemovalOutcome, Box<dyn std::error::Error>> {
    let credentials_path = pebble_config_home()?.join("credentials.json");
    let contents = match fs::read_to_string(&credentials_path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Ok(CredentialRemovalOutcome::Missing {
                path: credentials_path,
            });
        }
        Err(error) => return Err(Box::new(error)),
    };

    let mut parsed = serde_json::from_str::<serde_json::Value>(&contents)
        .unwrap_or_else(|_| serde_json::json!({}));
    if !parsed.is_object() {
        parsed = serde_json::json!({});
    }

    let removed = parsed
        .as_object_mut()
        .is_some_and(|object| object.remove(service.credential_key()).is_some());
    if !removed {
        return Ok(CredentialRemovalOutcome::Missing {
            path: credentials_path,
        });
    }

    fs::write(&credentials_path, serde_json::to_string_pretty(&parsed)?)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&credentials_path, fs::Permissions::from_mode(0o600))?;
    }

    Ok(CredentialRemovalOutcome::Removed {
        path: credentials_path,
    })
}

fn pebble_config_home() -> Result<PathBuf, Box<dyn std::error::Error>> {
    resolve_pebble_config_home()
        .ok_or_else(|| "could not resolve PEBBLE_CONFIG_HOME, HOME, or USERPROFILE".into())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LoginCommand {
    service: Option<AuthService>,
    api_key: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LogoutCommand {
    service: Option<AuthService>,
}

fn parse_auth_command(input: &str) -> Option<LoginCommand> {
    let mut parts = input.split_whitespace();
    let command = parts.next()?;
    if command != "/login" && command != "/auth" {
        return None;
    }
    parse_login_tokens(parts.collect::<Vec<_>>()).ok()
}

fn parse_logout_command(input: &str) -> Option<LogoutCommand> {
    let mut parts = input.split_whitespace();
    let command = parts.next()?;
    if command != "/logout" {
        return None;
    }
    parse_logout_tokens(parts.collect::<Vec<_>>()).ok()
}

fn parse_model_command(input: &str) -> Option<Option<String>> {
    let mut parts = input.split_whitespace();
    let command = parts.next()?;
    if command != "/model" && command != "/models" {
        return None;
    }

    let remainder = parts.collect::<Vec<_>>().join(" ");
    if remainder.is_empty() {
        Some(None)
    } else {
        Some(Some(remainder))
    }
}

fn parse_provider_command(input: &str) -> Option<Option<String>> {
    let mut parts = input.split_whitespace();
    let command = parts.next()?;
    if command != "/provider" && command != "/providers" {
        return None;
    }

    let remainder = parts.collect::<Vec<_>>().join(" ");
    if remainder.is_empty() {
        Some(None)
    } else {
        Some(Some(remainder))
    }
}

fn parse_proxy_command(input: &str) -> Option<Result<ProxyCommand, String>> {
    let mut parts = input.split_whitespace();
    let command = parts.next()?;
    if command != "/proxy" {
        return None;
    }

    let remainder = parts.collect::<Vec<_>>().join(" ");
    Some(parse_proxy_value(
        (!remainder.is_empty()).then_some(remainder.as_str()),
    ))
}

fn parse_reasoning_command(input: &str) -> Option<Result<Option<Option<ReasoningEffort>>, String>> {
    let mut parts = input.split_whitespace();
    let command = parts.next()?;
    if command != "/reasoning" && command != "/thinking" {
        return None;
    }

    let remainder = parts.collect::<Vec<_>>().join(" ");
    let trimmed = remainder.trim();
    Some(if trimmed.is_empty() {
        Ok(None)
    } else {
        parse_reasoning_effort_arg(trimmed).map(Some)
    })
}

fn parse_mode_command(input: &str) -> Option<Result<Option<CollaborationMode>, String>> {
    let mut parts = input.split_whitespace();
    let command = parts.next()?;
    if command != "/mode" {
        return None;
    }

    let remainder = parts.collect::<Vec<_>>().join(" ");
    let trimmed = remainder.trim();
    Some(if trimmed.is_empty() {
        Ok(None)
    } else {
        parse_collaboration_mode_arg(trimmed).map(Some)
    })
}

fn parse_fast_command(input: &str) -> Option<Result<Option<FastMode>, String>> {
    let mut parts = input.split_whitespace();
    let command = parts.next()?;
    if command != "/fast" {
        return None;
    }

    let remainder = parts.collect::<Vec<_>>().join(" ");
    let trimmed = remainder.trim();
    Some(match trimmed {
        "" => Ok(None),
        "on" => Ok(Some(FastMode::On)),
        "off" => Ok(Some(FastMode::Off)),
        other => Err(format!(
            "/fast accepts one optional argument: on or off (got {other})"
        )),
    })
}

fn parse_mcp_command(input: &str) -> Option<Result<McpCommand, String>> {
    let mut parts = input.split_whitespace();
    let command = parts.next()?;
    if command != "/mcp" {
        return None;
    }

    let args = parts.collect::<Vec<_>>();
    Some(match args.as_slice() {
        [] | ["status"] => Ok(McpCommand::Status),
        ["tools"] => Ok(McpCommand::Tools),
        ["reload"] => Ok(McpCommand::Reload),
        ["add", name] => Ok(McpCommand::Add {
            name: (*name).to_string(),
        }),
        ["enable", name] => Ok(McpCommand::Enable {
            name: (*name).to_string(),
        }),
        ["disable", name] => Ok(McpCommand::Disable {
            name: (*name).to_string(),
        }),
        [other, ..] => Err(format!(
            "/mcp accepts status, tools, reload, add <name>, enable <name>, or disable <name> (got {other})"
        )),
    })
}

fn parse_permissions_command(input: &str) -> Option<Result<Option<PermissionMode>, String>> {
    let mut parts = input.split_whitespace();
    let command = parts.next()?;
    if command == "/bypass" {
        let remainder = parts.collect::<Vec<_>>().join(" ");
        if !remainder.trim().is_empty() {
            return Some(Err("/bypass does not accept arguments".to_string()));
        }
        return Some(Ok(Some(PermissionMode::DangerFullAccess)));
    }
    if command != "/permissions" {
        return None;
    }

    let remainder = parts.collect::<Vec<_>>().join(" ");
    if remainder.trim().is_empty() {
        return Some(Ok(None));
    }
    Some(parse_permission_mode_arg(remainder.trim()).map(Some))
}

fn normalize_permission_mode(mode: &str) -> Option<&'static str> {
    match mode.trim().to_ascii_lowercase().as_str() {
        "read-only" | "readonly" | "read_only" => Some("read-only"),
        "workspace-write" | "workspacewrite" | "workspace_write" => Some("workspace-write"),
        "danger-full-access" | "dangerfullaccess" | "danger_full_access" => {
            Some("danger-full-access")
        }
        _ => None,
    }
}

fn permission_mode_from_label(mode: &str) -> PermissionMode {
    match mode {
        "read-only" => PermissionMode::ReadOnly,
        "workspace-write" => PermissionMode::WorkspaceWrite,
        "danger-full-access" => PermissionMode::DangerFullAccess,
        other => panic!("unsupported permission mode label: {other}"),
    }
}

fn parse_permission_mode_arg(value: &str) -> Result<PermissionMode, String> {
    normalize_permission_mode(value)
        .ok_or_else(|| {
            format!(
                "unsupported permission mode '{value}'. Use read-only, workspace-write, or danger-full-access."
            )
        })
        .map(permission_mode_from_label)
}

fn parse_collaboration_mode_arg(value: &str) -> Result<CollaborationMode, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "build" => Ok(CollaborationMode::Build),
        "plan" => Ok(CollaborationMode::Plan),
        other => Err(format!("unsupported mode '{other}'. Use build or plan.")),
    }
}

fn parse_reasoning_effort_arg(value: &str) -> Result<Option<ReasoningEffort>, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "default" | "auto" | "off" => Ok(None),
        "minimal" => Ok(Some(ReasoningEffort::Minimal)),
        "low" => Ok(Some(ReasoningEffort::Low)),
        "medium" | "on" => Ok(Some(ReasoningEffort::Medium)),
        "high" => Ok(Some(ReasoningEffort::High)),
        "xhigh" | "x-high" => Ok(Some(ReasoningEffort::XHigh)),
        other => Err(format!(
            "unsupported reasoning effort '{other}'. Use default, minimal, low, medium, high, or xhigh."
        )),
    }
}

fn default_permission_mode() -> PermissionMode {
    env::var("PEBBLE_PERMISSION_MODE")
        .ok()
        .as_deref()
        .and_then(normalize_permission_mode)
        .map_or(PermissionMode::WorkspaceWrite, permission_mode_from_label)
}

fn is_clear_provider_arg(value: &str) -> bool {
    matches!(value, "default" | "none" | "clear")
}

fn read_secret(prompt: &str) -> io::Result<String> {
    let mut stdout = io::stdout();
    write!(stdout, "{prompt}")?;
    stdout.flush()?;

    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        let mut buffer = String::new();
        io::stdin().read_line(&mut buffer)?;
        while matches!(buffer.chars().last(), Some('\n' | '\r')) {
            buffer.pop();
        }
        return Ok(buffer);
    }

    enable_raw_mode()?;
    if let Err(error) = execute!(stdout, EnableBracketedPaste) {
        let _ = disable_raw_mode();
        return Err(error);
    }
    let result = read_secret_raw(&mut stdout);
    let disable_paste_result = execute!(stdout, DisableBracketedPaste);
    let disable_raw_result = disable_raw_mode();
    writeln!(stdout)?;
    disable_paste_result?;
    disable_raw_result?;
    result
}

fn read_secret_raw(out: &mut impl Write) -> io::Result<String> {
    let mut secret = String::new();
    let opened_at = Instant::now();
    loop {
        match event::read()? {
            Event::Paste(data) => {
                secret.push_str(trim_trailing_line_endings(&data));
            }
            Event::Key(KeyEvent { kind, .. }) if kind == KeyEventKind::Release => {}
            Event::Key(KeyEvent {
                code: KeyCode::Enter,
                ..
            }) if should_ignore_stale_secret_submit(&secret, opened_at.elapsed()) => {}
            Event::Key(KeyEvent {
                code: KeyCode::Enter,
                ..
            }) => return Ok(secret),
            Event::Key(KeyEvent {
                code: KeyCode::Backspace,
                ..
            }) => {
                secret.pop();
            }
            Event::Key(KeyEvent {
                code: KeyCode::Char('c'),
                modifiers,
                ..
            }) if modifiers.contains(KeyModifiers::CONTROL) => {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "login cancelled",
                ));
            }
            Event::Key(KeyEvent {
                code: KeyCode::Char(ch),
                modifiers,
                ..
            }) if modifiers.is_empty() || modifiers == KeyModifiers::SHIFT => {
                secret.push(ch);
            }
            _ => {}
        }
        out.flush()?;
    }
}

fn should_ignore_stale_secret_submit(secret: &str, elapsed: Duration) -> bool {
    secret.is_empty() && elapsed <= SECRET_PROMPT_STALE_ENTER_WINDOW
}

fn trim_trailing_line_endings(value: &str) -> &str {
    value.trim_end_matches(['\r', '\n'])
}

fn base_runtime_tool_specs(tool_registry: &GlobalToolRegistry) -> Vec<RuntimeToolSpec> {
    tool_registry
        .entries()
        .iter()
        .map(|entry| RuntimeToolSpec {
            name: entry.definition.name.clone(),
            description: tuned_tool_description(
                &entry.definition.name,
                &entry
                    .definition
                    .description
                    .clone()
                    .unwrap_or_else(|| entry.definition.name.clone()),
            ),
            input_schema: entry.definition.input_schema.clone(),
            required_permission: entry.required_permission,
        })
        .collect()
}

fn tuned_tool_description(name: &str, base: &str) -> String {
    match name {
        "WebSearch" => format!(
            "{base} Prefer this for current information, release notes, changelogs, news, and finding relevant pages before reading them."
        ),
        "WebScrape" => format!(
            "{base} Prefer this when you already know the docs/article URLs and need readable page content or markdown to inspect."
        ),
        "WebFetch" => format!(
            "{base} Prefer this for a single known URL when you only need a quick fetch/summary; use WebScrape for richer doc/article reading."
        ),
        _ => base.to_string(),
    }
}

fn available_runtime_tool_specs(
    tool_registry: &GlobalToolRegistry,
    mcp_catalog: &McpCatalog,
) -> Vec<RuntimeToolSpec> {
    let mut specs = base_runtime_tool_specs(tool_registry);
    specs.extend(mcp_catalog.tool_specs());
    specs
}

fn filter_runtime_tool_specs(
    specs: Vec<RuntimeToolSpec>,
    allowed_tools: Option<&AllowedToolSet>,
) -> Vec<RuntimeToolSpec> {
    specs
        .into_iter()
        .filter(|spec| allowed_tools.is_none_or(|allowed| allowed.contains(&spec.name)))
        .collect()
}

fn load_mcp_catalog(cwd: &Path) -> Result<McpCatalog, Box<dyn std::error::Error>> {
    let config = ConfigLoader::default_for(cwd).load()?;
    let mut catalog = McpCatalog::default();
    let servers = configured_mcp_servers(&config)?;

    for (server_name, scoped) in servers {
        let mut status = McpServerStatus {
            server_name: server_name.clone(),
            scope: scoped.scope,
            enabled: scoped.enabled,
            transport: scoped.transport(),
            loaded: false,
            tool_count: 0,
            note: String::new(),
        };

        if !scoped.is_enabled() {
            status.note = "disabled in config".to_string();
            catalog.servers.push(status);
            continue;
        }

        match scoped.transport() {
            McpTransport::Stdio => match load_stdio_mcp_tools(&server_name, &scoped) {
                Ok(tools) => {
                    status.loaded = true;
                    status.tool_count = tools.len();
                    status.note = "stdio tools loaded".to_string();
                    catalog.tools.extend(tools);
                }
                Err(error) => {
                    status.note = format!("load failed: {error}");
                }
            },
            McpTransport::Http => match load_http_mcp_tools(&server_name, &scoped) {
                Ok(tools) => {
                    status.loaded = true;
                    status.tool_count = tools.len();
                    status.note = "http tools loaded".to_string();
                    catalog.tools.extend(tools);
                }
                Err(error) => {
                    status.note = format!("load failed: {error}");
                }
            },
            other => {
                status.note = format!(
                    "{:?} transport is configured but not executable in Pebble yet",
                    other
                );
            }
        }

        catalog.servers.push(status);
    }

    catalog
        .servers
        .sort_by(|left, right| left.server_name.cmp(&right.server_name));
    catalog
        .tools
        .sort_by(|left, right| left.exposed_name.cmp(&right.exposed_name));

    Ok(catalog)
}

fn configured_mcp_servers(
    config: &runtime::RuntimeConfig,
) -> Result<Vec<(String, ScopedMcpServerConfig)>, Box<dyn std::error::Error>> {
    let mut servers = config
        .mcp()
        .servers()
        .iter()
        .map(|(name, scoped)| (name.clone(), scoped.clone()))
        .collect::<Vec<_>>();

    servers.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(servers)
}

fn load_stdio_mcp_tools(
    server_name: &str,
    scoped: &ScopedMcpServerConfig,
) -> Result<Vec<McpToolBinding>, Box<dyn std::error::Error>> {
    let bootstrap = McpClientBootstrap::from_scoped_config(server_name, scoped);
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async move {
        let mut process = spawn_mcp_stdio_process(&bootstrap)?;
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(MCP_DISCOVERY_TIMEOUT_SECS),
            async {
                let initialize = process
                    .initialize(
                        JsonRpcId::Number(1),
                        McpInitializeParams {
                            protocol_version: "2025-03-26".to_string(),
                            capabilities: serde_json::json!({"roots": {}}),
                            client_info: McpInitializeClientInfo {
                                name: "pebble".to_string(),
                                version: VERSION.to_string(),
                            },
                        },
                    )
                    .await?;
                if let Some(error) = initialize.error {
                    return Err::<Vec<McpToolBinding>, Box<dyn std::error::Error>>(
                        format!("initialize failed: {}", error.message).into(),
                    );
                }
                process.send_initialized_notification().await?;

                let mut next_cursor = None;
                let mut bindings = Vec::new();
                let mut next_id = 2u64;
                loop {
                    let response = process
                        .list_tools(
                            JsonRpcId::Number(next_id),
                            Some(McpListToolsParams {
                                cursor: next_cursor.clone(),
                            }),
                        )
                        .await?;
                    next_id += 1;
                    if let Some(error) = response.error {
                        return Err(format!("tools/list failed: {}", error.message).into());
                    }
                    let Some(result) = response.result else {
                        break;
                    };
                    for tool in result.tools {
                        bindings.push(McpToolBinding {
                            exposed_name: mcp_tool_name(server_name, &tool.name),
                            server_name: server_name.to_string(),
                            upstream_name: tool.name,
                            description: tool.description.unwrap_or_else(|| "MCP tool".to_string()),
                            input_schema: tool
                                .input_schema
                                .unwrap_or_else(|| serde_json::json!({"type":"object"})),
                            config: scoped.clone(),
                        });
                    }
                    if result.next_cursor.is_none() {
                        break;
                    }
                    next_cursor = result.next_cursor;
                }

                Ok(bindings)
            },
        )
        .await;
        let _ = process.terminate().await;
        let _ = process.wait().await;

        match result {
            Ok(result) => result,
            Err(_) => Err::<Vec<McpToolBinding>, Box<dyn std::error::Error>>(
                format!(
                    "timed out after {}s during stdio MCP discovery",
                    MCP_DISCOVERY_TIMEOUT_SECS
                )
                .into(),
            ),
        }
    })
}

fn load_http_mcp_tools(
    server_name: &str,
    scoped: &ScopedMcpServerConfig,
) -> Result<Vec<McpToolBinding>, Box<dyn std::error::Error>> {
    let McpClientTransport::Http(transport) =
        McpClientBootstrap::from_scoped_config(server_name, scoped).transport
    else {
        return Err("server is not an HTTP MCP transport".into());
    };

    if transport.headers_helper.is_some() {
        return Err("headers_helper for remote MCP servers is not wired yet".into());
    }
    if transport.auth != McpClientAuth::None {
        return Err("OAuth-backed remote MCP servers are not wired yet".into());
    }

    let runtime = tokio::runtime::Runtime::new()?;
    let server_name = server_name.to_string();
    let scoped = scoped.clone();
    runtime.block_on(async move {
        let initialize = http_jsonrpc_request::<JsonValue>(
            &transport.url,
            &transport.headers,
            JsonRpcId::Number(1),
            "initialize",
            Some(serde_json::json!({
                "protocolVersion": "2025-03-26",
                "capabilities": {"roots": {}},
                "clientInfo": {"name": "pebble", "version": VERSION}
            })),
        )
        .await?;
        if let Some(error) = initialize.error {
            return Err::<Vec<McpToolBinding>, Box<dyn std::error::Error>>(
                format!("initialize failed: {}", error.message).into(),
            );
        }
        http_jsonrpc_notification(
            &transport.url,
            &transport.headers,
            "notifications/initialized",
            Some(serde_json::json!({})),
        )
        .await?;

        let mut next_cursor = None;
        let mut bindings = Vec::new();
        let mut next_id = 2_u64;
        loop {
            let response = http_jsonrpc_request::<McpListToolsResult>(
                &transport.url,
                &transport.headers,
                JsonRpcId::Number(next_id),
                "tools/list",
                Some(serde_json::to_value(McpListToolsParams {
                    cursor: next_cursor.clone(),
                })?),
            )
            .await?;
            next_id += 1;
            if let Some(error) = response.error {
                return Err(format!("tools/list failed: {}", error.message).into());
            }
            let Some(result) = response.result else {
                break;
            };
            for tool in result.tools {
                bindings.push(McpToolBinding {
                    exposed_name: mcp_tool_name(&server_name, &tool.name),
                    server_name: server_name.clone(),
                    upstream_name: tool.name,
                    description: tool.description.unwrap_or_else(|| "MCP tool".to_string()),
                    input_schema: tool
                        .input_schema
                        .unwrap_or_else(|| serde_json::json!({"type":"object"})),
                    config: scoped.clone(),
                });
            }
            if result.next_cursor.is_none() {
                break;
            }
            next_cursor = result.next_cursor;
        }

        Ok(bindings)
    })
}

fn call_mcp_tool(
    binding: &McpToolBinding,
    input: &JsonValue,
) -> Result<String, Box<dyn std::error::Error>> {
    match binding.config.transport() {
        McpTransport::Stdio => call_stdio_mcp_tool(binding, input),
        McpTransport::Http => call_http_mcp_tool(binding, input),
        other => Err(format!("MCP transport {:?} is not executable in Pebble yet", other).into()),
    }
}

fn call_stdio_mcp_tool(
    binding: &McpToolBinding,
    input: &JsonValue,
) -> Result<String, Box<dyn std::error::Error>> {
    let bootstrap = McpClientBootstrap::from_scoped_config(&binding.server_name, &binding.config);
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async move {
        let mut process = spawn_mcp_stdio_process(&bootstrap)?;
        let initialize = process
            .initialize(
                JsonRpcId::Number(1),
                McpInitializeParams {
                    protocol_version: "2025-03-26".to_string(),
                    capabilities: serde_json::json!({"roots": {}}),
                    client_info: McpInitializeClientInfo {
                        name: "pebble".to_string(),
                        version: VERSION.to_string(),
                    },
                },
            )
            .await?;
        if let Some(error) = initialize.error {
            let _ = process.terminate().await;
            let _ = process.wait().await;
            return Err::<String, Box<dyn std::error::Error>>(
                format!("initialize failed: {}", error.message).into(),
            );
        }
        process.send_initialized_notification().await?;

        let response = process
            .call_tool(
                JsonRpcId::Number(2),
                McpToolCallParams {
                    name: binding.upstream_name.clone(),
                    arguments: Some(input.clone()),
                    meta: None,
                },
            )
            .await?;
        let _ = process.terminate().await;
        let _ = process.wait().await;

        format_mcp_call_result(&binding.server_name, &binding.upstream_name, response)
    })
}

fn call_http_mcp_tool(
    binding: &McpToolBinding,
    input: &JsonValue,
) -> Result<String, Box<dyn std::error::Error>> {
    let McpClientTransport::Http(transport) =
        McpClientBootstrap::from_scoped_config(&binding.server_name, &binding.config).transport
    else {
        return Err("server is not an HTTP MCP transport".into());
    };
    if transport.headers_helper.is_some() {
        return Err("headers_helper for remote MCP servers is not wired yet".into());
    }
    if transport.auth != McpClientAuth::None {
        return Err("OAuth-backed remote MCP servers are not wired yet".into());
    }

    let runtime = tokio::runtime::Runtime::new()?;
    let server_name = binding.server_name.clone();
    let upstream_name = binding.upstream_name.clone();
    let input = input.clone();
    runtime.block_on(async move {
        let initialize = http_jsonrpc_request::<JsonValue>(
            &transport.url,
            &transport.headers,
            JsonRpcId::Number(1),
            "initialize",
            Some(serde_json::json!({
                "protocolVersion": "2025-03-26",
                "capabilities": {"roots": {}},
                "clientInfo": {"name": "pebble", "version": VERSION}
            })),
        )
        .await?;
        if let Some(error) = initialize.error {
            return Err::<String, Box<dyn std::error::Error>>(
                format!("initialize failed: {}", error.message).into(),
            );
        }
        http_jsonrpc_notification(
            &transport.url,
            &transport.headers,
            "notifications/initialized",
            Some(serde_json::json!({})),
        )
        .await?;

        let response = http_jsonrpc_request::<McpToolCallResult>(
            &transport.url,
            &transport.headers,
            JsonRpcId::Number(2),
            "tools/call",
            Some(serde_json::to_value(McpToolCallParams {
                name: upstream_name.clone(),
                arguments: Some(input),
                meta: None,
            })?),
        )
        .await?;
        format_mcp_call_result(&server_name, &upstream_name, response)
    })
}

async fn http_jsonrpc_request<TResult: serde::de::DeserializeOwned>(
    url: &str,
    headers: &std::collections::BTreeMap<String, String>,
    id: JsonRpcId,
    method: &str,
    params: Option<JsonValue>,
) -> Result<JsonRpcResponse<TResult>, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let mut request = client
        .post(url)
        .header("content-type", "application/json")
        .header("accept", "application/json, text/event-stream");
    for (key, value) in headers {
        request = request.header(
            HeaderName::from_bytes(key.as_bytes())?,
            HeaderValue::from_str(value)?,
        );
    }
    let response = request
        .json(&JsonRpcRequest::new(id, method.to_string(), params))
        .send()
        .await?;
    Ok(response.json::<JsonRpcResponse<TResult>>().await?)
}

async fn http_jsonrpc_notification(
    url: &str,
    headers: &std::collections::BTreeMap<String, String>,
    method: &str,
    params: Option<JsonValue>,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let mut request = client
        .post(url)
        .header("content-type", "application/json")
        .header("accept", "application/json, text/event-stream");
    for (key, value) in headers {
        request = request.header(
            HeaderName::from_bytes(key.as_bytes())?,
            HeaderValue::from_str(value)?,
        );
    }
    request
        .json(&serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params.unwrap_or_else(|| serde_json::json!({}))
        }))
        .send()
        .await?;
    Ok(())
}

fn format_mcp_call_result(
    server_name: &str,
    upstream_name: &str,
    response: JsonRpcResponse<McpToolCallResult>,
) -> Result<String, Box<dyn std::error::Error>> {
    if let Some(error) = response.error {
        return Err(format!("tools/call failed: {}", error.message).into());
    }
    let Some(result) = response.result else {
        return Err("tools/call returned no result".into());
    };
    Ok(serde_json::to_string_pretty(&serde_json::json!({
        "server": server_name,
        "tool": upstream_name,
        "content": result.content,
        "structuredContent": result.structured_content,
        "isError": result.is_error.unwrap_or(false),
    }))?)
}

fn print_mcp_status(catalog: &McpCatalog) {
    if catalog.servers.is_empty() {
        println!("{}", report_title("MCP"));
        println!("  {} {}", report_label("servers:"), 0);
        println!("  {} {}", report_label("tools:"), 0);
        println!(
            "  {} add `mcpServers` to `.pebble/settings.json` to expose MCP tools",
            report_label("hint:")
        );
        return;
    }

    println!("{}", report_title("MCP"));
    println!("  {} {}", report_label("servers:"), catalog.servers.len());
    println!("  {} {}", report_label("tools:"), catalog.tools.len());
    println!();
    for server in &catalog.servers {
        println!("  {}", format!("{}", server.server_name.as_str().bold()));
        println!(
            "    {} {}  {} {:?}  {} {}  {} {}",
            report_label("scope"),
            config_source_label(server.scope),
            report_label("transport"),
            server.transport,
            report_label("tools"),
            server.tool_count,
            report_label("status"),
            if !server.enabled {
                "disabled"
            } else if server.loaded {
                "ready"
            } else {
                "unavailable"
            }
        );
        println!("    {} {}", report_label("note"), server.note);
    }
}

fn print_mcp_tools(catalog: &McpCatalog) {
    if catalog.tools.is_empty() {
        print_mcp_status(catalog);
        return;
    }

    println!("{}", report_title("MCP Tools"));
    println!("  {} {}", report_label("count:"), catalog.tools.len());
    println!();
    for tool in &catalog.tools {
        println!("  {}", format!("{}", tool.exposed_name.as_str().bold()));
        println!(
            "    {} {}\n    {} {}\n    {} {}",
            report_label("upstream"),
            tool.upstream_name,
            report_label("server"),
            tool.server_name,
            report_label("description"),
            tool.description
        );
    }
}

fn config_source_label(source: ConfigSource) -> &'static str {
    match source {
        ConfigSource::User => "user",
        ConfigSource::Project => "project",
        ConfigSource::Local => "local",
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StatusContext {
    cwd: PathBuf,
    session_path: Option<PathBuf>,
    loaded_config_files: usize,
    discovered_config_files: usize,
    instruction_file_count: usize,
    memory_file_count: usize,
    project_root: Option<PathBuf>,
    git_branch: Option<String>,
    sandbox_summary: String,
    web_tools_summary: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StatusUsage {
    message_count: usize,
    turns: u32,
    latest: TokenUsage,
    cumulative: TokenUsage,
    estimated_tokens: usize,
}

#[derive(Debug, Clone)]
struct ResumeCommandOutcome {
    session: Session,
    message: Option<String>,
}

fn status_context(
    session_path: Option<&Path>,
) -> Result<StatusContext, Box<dyn std::error::Error>> {
    let cwd = env::current_dir()?;
    let loader = ConfigLoader::default_for(&cwd);
    let discovered_config_files = loader.discover().len();
    let runtime_config = loader.load()?;
    let sandbox_status = resolve_sandbox_status(runtime_config.sandbox(), &cwd);
    let project_context = runtime::ProjectContext::discover_with_git(&cwd, DEFAULT_DATE)?;
    let (project_root, git_branch) =
        parse_git_status_metadata(project_context.git_status.as_deref());
    Ok(StatusContext {
        cwd,
        session_path: session_path.map(Path::to_path_buf),
        loaded_config_files: runtime_config.loaded_entries().len(),
        discovered_config_files,
        instruction_file_count: project_context.instruction_files.len(),
        memory_file_count: project_context.memory_files.len(),
        project_root,
        git_branch,
        sandbox_summary: format_sandbox_status(&sandbox_status),
        web_tools_summary: format_web_tools_status(),
    })
}

fn parse_git_status_metadata(status: Option<&str>) -> (Option<PathBuf>, Option<String>) {
    let Some(status) = status else {
        return (None, None);
    };
    let mut branch = None;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("## ") {
            branch = Some(rest.split("...").next().unwrap_or(rest).trim().to_string());
            break;
        }
    }
    let project_root = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|stdout| PathBuf::from(stdout.trim()));
    (project_root, branch)
}

fn report_title(text: &str) -> String {
    format!("{}", text.bold().with(Color::Yellow))
}

fn report_section(text: &str) -> String {
    format!("{}", text.bold().with(Color::Cyan))
}

fn report_label(text: &str) -> String {
    format!("{}", text.with(Color::DarkGrey))
}

fn report_value(text: impl std::fmt::Display) -> String {
    text.to_string()
}

fn format_status_report(
    service: ApiService,
    model: &str,
    usage: StatusUsage,
    permission_mode: &str,
    provider: Option<&str>,
    proxy_tool_calls: bool,
    collaboration_mode: CollaborationMode,
    reasoning_effort: Option<ReasoningEffort>,
    fast_mode: FastMode,
    mcp_catalog: &McpCatalog,
    context: &StatusContext,
) -> String {
    let provider_line = provider
        .map(|provider| {
            format!(
                "\n    {} {}",
                report_label("provider"),
                report_value(provider)
            )
        })
        .unwrap_or_default();
    [
        format!(
            "{}\n  {}\n    {} {}\n    {} {}{}\n    {} {}\n    {} {}\n    {} {}\n    {} {}\n    {} {}\n    {} {}\n    {} {}\n    {} {}",
            report_title("Pebble Status"),
            report_section("Session"),
            report_label("service"),
            report_value(service.display_name()),
            report_label("model"),
            report_value(model),
            provider_line,
            report_label("permission_mode"),
            report_value(permission_mode),
            report_label("proxy_tools"),
            if proxy_tool_calls { "enabled" } else { "disabled" },
            report_label("mode"),
            collaboration_mode.as_str(),
            report_label("reasoning"),
            reasoning_effort_label(reasoning_effort),
            report_label("fast_mode"),
            fast_mode.as_str(),
            report_label("messages"),
            usage.message_count,
            report_label("turns"),
            usage.turns,
            report_label("estimated_tokens"),
            usage.estimated_tokens,
        ),
        format!(
            "  {}\n    {} {}\n    {} {}\n    {} {}\n    {} {}",
            report_section("Usage"),
            report_label("latest_total"),
            usage.latest.total_tokens(),
            report_label("cumulative_input"),
            usage.cumulative.input_tokens,
            report_label("cumulative_output"),
            usage.cumulative.output_tokens,
            report_label("cumulative_total"),
            usage.cumulative.total_tokens(),
        ),
        format!(
            "  {}\n    {} {}\n    {} {}\n    {} {}\n    {} {}\n    {} loaded {}/{}\n    {} {}\n    {} {}\n    {} servers={} tools={}",
            report_section("Workspace"),
            report_label("cwd"),
            context.cwd.display(),
            report_label("project_root"),
            context
                .project_root
                .as_ref()
                .map_or_else(|| "unknown".to_string(), |path| path.display().to_string()),
            report_label("git_branch"),
            context.git_branch.as_deref().unwrap_or("unknown"),
            report_label("session"),
            context.session_path.as_ref().map_or_else(
                || "live-repl".to_string(),
                |path| path.display().to_string()
            ),
            report_label("config_files"),
            context.loaded_config_files,
            context.discovered_config_files,
            report_label("instruction_files"),
            context.instruction_file_count,
            report_label("memory_files"),
            context.memory_file_count,
            report_label("mcp"),
            mcp_catalog.servers.len(),
            mcp_catalog.tools.len(),
        ),
        format!(
            "  {}\n    {}",
            report_section("Sandbox"),
            context.sandbox_summary.replace('\n', "\n    ")
        ),
        format!(
            "  {}\n    {}",
            report_section("Web Tools"),
            context.web_tools_summary.replace('\n', "\n    ")
        ),
    ]
    .join("\n\n")
}

fn format_web_tools_status() -> String {
    let api_key_configured = resolve_exa_api_key().is_ok();
    let base_url = resolve_exa_base_url();
    let (web_search_available, web_scrape_available) = current_tool_registry()
        .map(|registry| {
            let mut has_search = false;
            let mut has_scrape = false;
            for entry in registry.entries() {
                if entry.definition.name == "WebSearch" {
                    has_search = true;
                }
                if entry.definition.name == "WebScrape" {
                    has_scrape = true;
                }
            }
            (has_search, has_scrape)
        })
        .unwrap_or((false, false));

    format!(
        "service=Exa\nbase_url={base_url}\napi_key={}\nweb_search={}\nweb_scrape={}",
        if api_key_configured {
            "configured"
        } else {
            "missing"
        },
        if web_search_available {
            "available"
        } else {
            "missing"
        },
        if web_scrape_available {
            "available"
        } else {
            "missing"
        },
    )
}

fn resolve_exa_base_url() -> String {
    env::var("EXA_BASE_URL").unwrap_or_else(|_| "https://api.exa.ai".to_string())
}

fn resolve_exa_api_key() -> Result<String, Box<dyn std::error::Error>> {
    match env::var("EXA_API_KEY") {
        Ok(value) if !value.trim().is_empty() => Ok(value),
        Ok(_) => Err("EXA_API_KEY is empty".into()),
        Err(env::VarError::NotPresent) => {
            let path = pebble_config_home()?.join("credentials.json");
            let contents = fs::read_to_string(path)?;
            let parsed = serde_json::from_str::<serde_json::Value>(&contents)?;
            parsed
                .get("exa_api_key")
                .and_then(serde_json::Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .map(ToOwned::to_owned)
                .ok_or_else(|| "missing exa_api_key".into())
        }
        Err(error) => Err(Box::new(error)),
    }
}

fn format_resume_report(session_path: &str, message_count: usize, turns: u32) -> String {
    format!(
        "Session resumed\n  Session file     {session_path}\n  Messages         {message_count}\n  Turns            {turns}"
    )
}

fn format_sandbox_status(status: &runtime::SandboxStatus) -> String {
    let mode = status.filesystem_mode.as_str();
    let active = if status.active { "active" } else { "inactive" };
    let network = if status.network_active {
        "isolated"
    } else if status.requested.network_isolation {
        "requested-unavailable"
    } else {
        "shared"
    };
    let namespace = if status.namespace_active {
        "restricted"
    } else if status.requested.namespace_restrictions {
        "requested-unavailable"
    } else {
        "shared"
    };
    let mounts = if status.allowed_mounts.is_empty() {
        "<none>".to_string()
    } else {
        status.allowed_mounts.join(", ")
    };
    let mut line = format!(
        "Enabled          {}\n  Status           {}\n  Namespace        {}\n  Network          {}\n  Filesystem       {}\n  Allowed mounts   {}\n  Container        {}",
        if status.enabled { "yes" } else { "no" },
        active,
        namespace,
        network,
        mode,
        mounts,
        if status.in_container { "yes" } else { "no" },
    );
    if let Some(reason) = &status.fallback_reason {
        let _ = write!(line, "\n  Fallback         {reason}");
    }
    line
}

fn format_auto_compaction_notice(removed: usize) -> String {
    format!("[auto-compacted: removed {removed} messages]")
}

fn sessions_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let cwd = env::current_dir()?;
    let path = cwd.join(".pebble").join("sessions");
    fs::create_dir_all(&path)?;
    Ok(path)
}

fn create_managed_session_handle() -> Result<SessionHandle, Box<dyn std::error::Error>> {
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

fn resolve_session_reference(reference: &str) -> Result<SessionHandle, Box<dyn std::error::Error>> {
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

fn list_managed_sessions() -> Result<Vec<ManagedSessionSummary>, Box<dyn std::error::Error>> {
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
            model: stored.map(|metadata| metadata.model.clone()),
            started_at: stored.map(|metadata| metadata.started_at.clone()),
            last_prompt: stored.and_then(|metadata| metadata.last_prompt.clone()),
        });
    }
    sessions.sort_by(|left, right| right.modified_epoch_secs.cmp(&left.modified_epoch_secs));
    Ok(sessions)
}

fn render_session_list(active_session_id: &str) -> Result<String, Box<dyn std::error::Error>> {
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
        lines.push(format!(
            "  {id:<20} {marker:<10} msgs={msgs:<4} model={model:<24} started={started} modified={modified} last={last_prompt} path={path}",
            id = session.id,
            msgs = session.message_count,
            model = model,
            started = started,
            modified = session.modified_epoch_secs,
            last_prompt = last_prompt,
            path = session.path.display(),
        ));
    }
    Ok(lines.join("\n"))
}

fn truncate_for_summary(text: &str, max_chars: usize) -> String {
    let trimmed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = trimmed.chars();
    let truncated = chars.by_ref().take(max_chars).collect::<String>();
    if chars.next().is_some() {
        format!("{truncated}…")
    } else {
        truncated
    }
}

fn summarize_tool_payload(payload: &str) -> String {
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

fn current_epoch_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().try_into().unwrap_or(u64::MAX))
        .unwrap_or_default()
}

fn current_timestamp_rfc3339ish() -> String {
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

fn derive_session_metadata(
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
    }
}

fn session_runtime_state(
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
    let model = metadata
        .map(|metadata| metadata.model.clone())
        .unwrap_or_else(|| fallback_model.to_string());
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
    let fast_mode = metadata
        .and_then(|metadata| metadata.fast_mode)
        .map(|enabled| if enabled { FastMode::On } else { FastMode::Off })
        .unwrap_or(fallback_fast_mode);
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

fn session_age_secs(modified_epoch_secs: u64) -> u64 {
    current_epoch_secs().saturating_sub(modified_epoch_secs)
}

fn auto_compact_inactive_sessions(
    active_session_id: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    for summary in list_managed_sessions()? {
        if summary.id == active_session_id
            || session_age_secs(summary.modified_epoch_secs) < OLD_SESSION_COMPACTION_AGE_SECS
        {
            continue;
        }
        let path = summary.path.clone();
        let Ok(session) = Session::load_from_path(&path) else {
            continue;
        };
        if !runtime::should_compact(&session, CompactionConfig::default()) {
            continue;
        }
        let mut compacted =
            runtime::compact_session(&session, CompactionConfig::default()).compacted_session;
        let model = compacted.metadata.as_ref().map_or_else(
            || DEFAULT_MODEL.to_string(),
            |metadata| metadata.model.clone(),
        );
        let state = session_runtime_state(
            &compacted,
            &model,
            None,
            default_permission_mode(),
            CollaborationMode::Build,
            None,
            FastMode::Off,
            false,
        );
        compacted.metadata = Some(derive_session_metadata(
            &compacted,
            &model,
            state.allowed_tools.as_ref(),
            state.permission_mode,
            state.collaboration_mode,
            state.reasoning_effort,
            state.fast_mode,
            state.proxy_tool_calls,
        ));
        compacted.save_to_path(&path)?;
    }
    Ok(())
}

fn render_config_report(section: Option<&str>) -> Result<String, Box<dyn std::error::Error>> {
    let cwd = env::current_dir()?;
    let loader = ConfigLoader::default_for(&cwd);
    let discovered = loader.discover();
    let runtime_config = loader.load()?;

    let mut lines = vec![
        format!(
            "Config\n  Working directory {}\n  Loaded files      {}\n  Merged keys       {}",
            cwd.display(),
            runtime_config.loaded_entries().len(),
            runtime_config.merged().len()
        ),
        "Discovered files".to_string(),
    ];
    for entry in discovered {
        let source = config_source_label(entry.source);
        let status = if runtime_config
            .loaded_entries()
            .iter()
            .any(|loaded_entry| loaded_entry.path == entry.path)
        {
            "loaded"
        } else {
            "missing"
        };
        lines.push(format!(
            "  {source:<7} {status:<7} {}",
            entry.path.display()
        ));
    }

    if let Some(section) = section {
        lines.push(format!("Merged section: {section}"));
        let value = match section {
            "env" => runtime_config.get("env"),
            "hooks" => runtime_config.get("hooks"),
            "model" => runtime_config.get("model"),
            "plugins" => runtime_config.get("plugins"),
            other => {
                lines.push(format!(
                    "  Unsupported config section '{other}'. Use env, hooks, model, or plugins."
                ));
                return Ok(lines.join("\n"));
            }
        };
        lines.push(format!(
            "  {}",
            match value {
                Some(value) => value.render(),
                None => "<unset>".to_string(),
            }
        ));
        return Ok(lines.join("\n"));
    }

    lines.push("Merged JSON".to_string());
    lines.push(format!("  {}", runtime_config.as_json().render()));
    Ok(lines.join("\n"))
}

fn current_plugin_manager() -> Result<PluginManager, Box<dyn std::error::Error>> {
    let cwd = env::current_dir()?;
    let loader = ConfigLoader::default_for(&cwd);
    let runtime_config = loader.load()?;
    Ok(build_plugin_manager(&cwd, &loader, &runtime_config))
}

fn render_plugins_report(plugins: &[PluginSummary]) -> String {
    let mut lines = vec![
        report_title("Plugins"),
        format!("  {} {}", report_label("count:"), plugins.len()),
    ];
    if plugins.is_empty() {
        lines.push(format!(
            "  {} no plugins discovered",
            report_label("state:")
        ));
        return lines.join("\n");
    }

    for plugin in plugins {
        lines.push(String::new());
        lines.push(format!(
            "  {}",
            format!("{}", plugin.metadata.name.as_str().bold())
        ));
        lines.push(format!(
            "    {} {}  {} {}  {} {}  {} {}",
            report_label("id"),
            plugin.metadata.id,
            report_label("kind"),
            plugin.metadata.kind,
            report_label("version"),
            plugin.metadata.version,
            report_label("state"),
            if plugin.enabled {
                "enabled"
            } else {
                "disabled"
            }
        ));
        lines.push(format!("    {}", plugin.metadata.description));
        if let Some(root) = &plugin.metadata.root {
            lines.push(format!("    {} {}", report_label("root"), root.display()));
        }
    }

    lines.join("\n")
}

fn resolve_plugin_summary(
    manager: &PluginManager,
    target: &str,
) -> Result<PluginSummary, PluginError> {
    let plugins = manager.list_installed_plugins()?;
    plugins
        .into_iter()
        .find(|plugin| plugin.metadata.id == target || plugin.metadata.name == target)
        .ok_or_else(|| PluginError::NotFound(format!("plugin `{target}` was not found")))
}

fn render_plugin_action_result(
    action: &str,
    plugin_id: &str,
    name: &str,
    version_line: &str,
    state: &str,
) -> String {
    format!(
        "{}\n  {} {}\n  {} {}\n  {} {}\n  {} {}\n  {} {}",
        report_title("Plugins"),
        report_label("action:"),
        action,
        report_label("plugin:"),
        name,
        report_label("id:"),
        plugin_id,
        report_label("version:"),
        version_line,
        report_label("state:"),
        state
    )
}

fn handle_plugins_command(
    action: Option<&str>,
    target: Option<&str>,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut manager = current_plugin_manager()?;
    match action {
        None | Some("list") => Ok(render_plugins_report(&manager.list_installed_plugins()?)),
        Some("help") => Ok([
            "Plugins".to_string(),
            "  Usage            /plugins [list|help|install <path>|enable <id>|disable <id>|uninstall <id>|update <id>]".to_string(),
            "  Install          Point at a local plugin root that contains `.codex-plugin/plugin.json`.".to_string(),
            "  Example          /plugins install ./plugins/my-plugin".to_string(),
            "  Enable           /plugins enable <id>".to_string(),
            "  Disable          /plugins disable <id>".to_string(),
            "  Layout           Local plugins typically store skills, optional MCP manifests, and plugin metadata under the plugin root.".to_string(),
        ]
        .join("\n")),
        Some("install") => {
            let Some(target) = target else {
                return Ok("Plugins\n  error: missing install target\n  usage: /plugins install <path>".to_string());
            };
            let install = manager.install(target)?;
            let plugin = resolve_plugin_summary(&manager, &install.plugin_id).ok();
            Ok(render_plugin_action_result(
                "installed",
                &install.plugin_id,
                plugin
                    .as_ref()
                    .map(|plugin| plugin.metadata.name.as_str())
                    .unwrap_or(install.plugin_id.as_str()),
                &install.version,
                if plugin.as_ref().is_some_and(|plugin| plugin.enabled) {
                    "enabled"
                } else {
                    "disabled"
                },
            ))
        }
        Some("enable") => {
            let Some(target) = target else {
                return Ok("Plugins\n  error: missing enable target\n  usage: /plugins enable <id>".to_string());
            };
            let plugin = resolve_plugin_summary(&manager, target)?;
            manager.enable(&plugin.metadata.id)?;
            Ok(render_plugin_action_result(
                "enabled",
                &plugin.metadata.id,
                &plugin.metadata.name,
                &plugin.metadata.version,
                "enabled",
            ))
        }
        Some("disable") => {
            let Some(target) = target else {
                return Ok("Plugins\n  error: missing disable target\n  usage: /plugins disable <id>".to_string());
            };
            let plugin = resolve_plugin_summary(&manager, target)?;
            manager.disable(&plugin.metadata.id)?;
            Ok(render_plugin_action_result(
                "disabled",
                &plugin.metadata.id,
                &plugin.metadata.name,
                &plugin.metadata.version,
                "disabled",
            ))
        }
        Some("uninstall") => {
            let Some(target) = target else {
                return Ok("Plugins\n  error: missing uninstall target\n  usage: /plugins uninstall <id>".to_string());
            };
            let plugin = resolve_plugin_summary(&manager, target)?;
            manager.uninstall(&plugin.metadata.id)?;
            Ok(format!(
                "Plugins\n  action:  uninstalled\n  plugin:  {}\n  id:      {}",
                plugin.metadata.name, plugin.metadata.id
            ))
        }
        Some("update") => {
            let Some(target) = target else {
                return Ok("Plugins\n  error: missing update target\n  usage: /plugins update <id>".to_string());
            };
            let plugin = resolve_plugin_summary(&manager, target)?;
            let update = manager.update(&plugin.metadata.id)?;
            Ok(render_plugin_action_result(
                "updated",
                &update.plugin_id,
                &plugin.metadata.name,
                &format!("{} -> {}", update.old_version, update.new_version),
                if resolve_plugin_summary(&manager, &update.plugin_id)
                    .ok()
                    .is_some_and(|summary| summary.enabled)
                {
                    "enabled"
                } else {
                    "disabled"
                },
            ))
        }
        Some(other) => Ok(format!(
            "Plugins\n  error: unsupported action `{other}`\n  usage: /plugins [list|install|enable|disable|uninstall|update]"
        )),
    }
}

fn plugins_command_is_mutating(action: Option<&str>) -> bool {
    matches!(
        action.unwrap_or("list"),
        "install" | "enable" | "disable" | "uninstall" | "update"
    )
}

fn render_memory_report() -> Result<String, Box<dyn std::error::Error>> {
    let cwd = env::current_dir()?;
    let project_context = runtime::ProjectContext::discover(&cwd, DEFAULT_DATE)?;
    let mut lines = vec![format!(
        "Memory\n  Working directory {}\n  Instruction files {}\n  Memory files      {}",
        cwd.display(),
        project_context.instruction_files.len(),
        project_context.memory_files.len(),
    )];

    lines.push("Instruction files".to_string());
    if project_context.instruction_files.is_empty() {
        lines.push("  No instruction markdown files discovered.".to_string());
    } else {
        for (index, file) in project_context.instruction_files.iter().enumerate() {
            let preview = file.content.lines().next().unwrap_or("").trim();
            lines.push(format!("  {}. {}", index + 1, file.path.display()));
            lines.push(format!(
                "     lines={} preview={}",
                file.content.lines().count(),
                if preview.is_empty() {
                    "<empty>"
                } else {
                    preview
                }
            ));
        }
    }

    lines.push("Memory files".to_string());
    if project_context.memory_files.is_empty() {
        lines.push("  No `.pebble/memory` files discovered.".to_string());
    } else {
        for (index, file) in project_context.memory_files.iter().enumerate() {
            let preview = file.content.lines().next().unwrap_or("").trim();
            lines.push(format!("  {}. {}", index + 1, file.path.display()));
            lines.push(format!(
                "     lines={} preview={}",
                file.content.lines().count(),
                if preview.is_empty() {
                    "<empty>"
                } else {
                    preview
                }
            ));
        }
    }

    Ok(lines.join("\n"))
}

fn render_diff_report() -> Result<String, Box<dyn std::error::Error>> {
    let cwd = env::current_dir()?;
    let staged = run_git_capture(&cwd, &["diff", "--cached"])?;
    let unstaged = run_git_capture(&cwd, &["diff"])?;

    let mut sections = Vec::new();
    if !staged.trim().is_empty() {
        sections.push(format!("Staged changes:\n{}", staged.trim_end()));
    }
    if !unstaged.trim().is_empty() {
        sections.push(format!("Unstaged changes:\n{}", unstaged.trim_end()));
    }

    if sections.is_empty() {
        return Ok(
            "Diff\n  Result           clean working tree\n  Detail           no current changes"
                .to_string(),
        );
    }

    Ok(format!("Diff\n\n{}", sections.join("\n\n")))
}

fn run_git_capture(cwd: &Path, args: &[&str]) -> Result<String, Box<dyn std::error::Error>> {
    let output = Command::new("git").args(args).current_dir(cwd).output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(format!("git {} failed: {stderr}", args.join(" ")).into());
    }
    Ok(String::from_utf8(output.stdout)?)
}

fn render_version_report() -> String {
    let target = BUILD_TARGET.unwrap_or("unknown");
    format!(
        "Version\n  Version          {VERSION}\n  Target           {target}\n  Build date       {DEFAULT_DATE}"
    )
}

fn render_export_text(session: &Session) -> String {
    let mut lines = vec!["# Conversation Export".to_string(), String::new()];
    for (index, message) in session.messages.iter().enumerate() {
        let role = match message.role {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Tool => "tool",
        };
        lines.push(format!("## {}. {role}", index + 1));
        for block in &message.blocks {
            match block {
                ContentBlock::Text { text } => lines.push(text.clone()),
                ContentBlock::Thinking { text, signature } => {
                    lines.push(format!(
                        "[thinking hidden chars={} signature={}]",
                        text.chars().count(),
                        if signature.is_some() {
                            "present"
                        } else {
                            "absent"
                        }
                    ));
                }
                ContentBlock::ToolUse { id, name, input } => {
                    lines.push(format!("[tool_use id={id} name={name}] {input}"));
                }
                ContentBlock::ToolResult {
                    tool_use_id,
                    tool_name,
                    output,
                    is_error,
                } => {
                    lines.push(format!(
                        "[tool_result id={tool_use_id} name={tool_name} error={is_error}] {output}"
                    ));
                }
            }
        }
        lines.push(String::new());
    }
    lines.join("\n")
}

fn assistant_text_from_messages(messages: &[ConversationMessage]) -> String {
    messages
        .iter()
        .flat_map(|message| message.blocks.iter())
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

fn default_export_filename(session: &Session) -> String {
    let stem = session
        .messages
        .iter()
        .find_map(|message| match message.role {
            MessageRole::User => message.blocks.iter().find_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            }),
            _ => None,
        })
        .map_or("conversation", |text| {
            text.lines().next().unwrap_or("conversation")
        })
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .split('-')
        .filter(|part| !part.is_empty())
        .take(8)
        .collect::<Vec<_>>()
        .join("-");
    let fallback = if stem.is_empty() {
        "conversation"
    } else {
        &stem
    };
    format!("{fallback}.txt")
}

fn resolve_export_path(
    requested_path: Option<&str>,
    session: &Session,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let cwd = env::current_dir()?;
    let file_name =
        requested_path.map_or_else(|| default_export_filename(session), ToOwned::to_owned);
    let final_name = if Path::new(&file_name)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("txt"))
    {
        file_name
    } else {
        format!("{file_name}.txt")
    };
    Ok(cwd.join(final_name))
}

fn render_repl_help() -> String {
    let intro = ui::panel(
        "Pebble REPL",
        &[
            ui::PanelRow::Line(
                "An agentic coding shell. Slash-commands control the session; anything else is forwarded to the model."
                    .to_string(),
            ),
            ui::PanelRow::Blank,
            ui::PanelRow::Section("Tips".to_string()),
            ui::PanelRow::Line(
                "• /login and /auth open a service picker when no service is provided".to_string(),
            ),
            ui::PanelRow::Line(
                "• /logout removes saved credentials for a service".to_string(),
            ),
            ui::PanelRow::Line(
                "• supported auth services: nanogpt, synthetic, openai-codex, opencode-go, exa"
                    .to_string(),
            ),
            ui::PanelRow::Line(
                "• /provider is only available for NanoGPT-backed models".to_string(),
            ),
        ],
    );
    format!("{intro}\n\n{}", render_slash_command_help())
}

fn repl_completion_candidates(cli: &LiveCli) -> Vec<String> {
    let mut candidates = BTreeSet::new();

    for candidate in command_names_and_aliases() {
        candidates.insert(candidate);
    }

    for topic in ["help", "auth", "sessions", "extensions", "web"] {
        candidates.insert(format!("/help {topic}"));
    }

    for service in ["nanogpt", "synthetic", "openai-codex", "opencode-go", "exa"] {
        candidates.insert(format!("/login {service}"));
        candidates.insert(format!("/auth {service}"));
        candidates.insert(format!("/logout {service}"));
    }

    for mode in ["read-only", "workspace-write", "danger-full-access"] {
        candidates.insert(format!("/permissions {mode}"));
    }
    candidates.insert("/bypass".to_string());

    for value in ["on", "off"] {
        candidates.insert(format!("/thinking {value}"));
        candidates.insert(format!("/proxy {value}"));
    }
    candidates.insert("/proxy status".to_string());

    candidates.insert("/mcp status".to_string());
    candidates.insert("/mcp tools".to_string());
    candidates.insert("/mcp reload".to_string());
    candidates.insert("/mcp add".to_string());
    candidates.insert("/mcp enable".to_string());
    candidates.insert("/mcp disable".to_string());

    candidates.insert("/branch list".to_string());
    candidates.insert("/branch create".to_string());
    candidates.insert("/branch switch".to_string());

    candidates.insert("/worktree list".to_string());
    candidates.insert("/worktree add".to_string());
    candidates.insert("/worktree remove".to_string());
    candidates.insert("/worktree prune".to_string());

    candidates.insert("/plugins help".to_string());
    candidates.insert("/plugins list".to_string());
    candidates.insert("/plugins install".to_string());
    candidates.insert("/plugins enable".to_string());
    candidates.insert("/plugins disable".to_string());
    candidates.insert("/plugins uninstall".to_string());
    candidates.insert("/plugins update".to_string());

    candidates.insert("/skills list".to_string());
    candidates.insert("/skills help".to_string());
    candidates.insert("/skills init".to_string());
    candidates.insert("/agents list".to_string());
    candidates.insert("/agents help".to_string());
    candidates.insert("/session list".to_string());
    candidates.insert("/session switch".to_string());
    candidates.insert("/resume last".to_string());

    for model in model_completion_candidates(&cli.model) {
        candidates.insert(format!("/model {model}"));
    }

    if cli.service == ApiService::NanoGpt {
        candidates.insert("/provider default".to_string());
        if let Some(provider) = provider_for_model(&cli.model) {
            candidates.insert(format!("/provider {provider}"));
        }
    }

    if let Ok(sessions) = list_managed_sessions() {
        for session in sessions {
            candidates.insert(format!("/resume {}", session.id));
            candidates.insert(format!("/resume {}", session.path.display()));
            candidates.insert(format!("/session switch {}", session.id));
        }
    }

    if let Ok(entries) = fs::read_dir(env::current_dir().unwrap_or_else(|_| PathBuf::from("."))) {
        for entry in entries.flatten().take(64) {
            let path = entry.path();
            let display = path.display().to_string();
            candidates.insert(format!("/export {display}"));
            candidates.insert(format!("/plugins install {display}"));
            if path.is_dir() {
                candidates.insert(format!("/worktree add {display}"));
                candidates.insert(format!("/worktree remove {display}"));
            }
        }
    }

    candidates.into_iter().collect()
}

fn model_completion_candidates(current_model: &str) -> Vec<String> {
    let mut candidates = BTreeSet::new();
    for alias in [
        "default",
        "glm",
        "glm5",
        "glm-5",
        "glm5.1",
        "glm-5.1",
        "zai-org/glm-5",
        "zai-org/glm-5.1",
    ] {
        candidates.insert(alias.to_string());
    }

    candidates.insert(DEFAULT_MODEL.to_string());
    candidates.insert(current_model.to_string());

    if let Ok(state) = load_model_state() {
        if let Some(model) = state.current_model {
            candidates.insert(model);
        }
        for favorite in state.favorite_models {
            candidates.insert(favorite);
        }
    }

    candidates.into_iter().collect()
}

impl McpCatalog {
    fn tool_specs(&self) -> Vec<RuntimeToolSpec> {
        self.tools
            .iter()
            .map(|tool| RuntimeToolSpec {
                name: tool.exposed_name.clone(),
                description: format!(
                    "MCP {}::{} - {}",
                    tool.server_name, tool.upstream_name, tool.description
                ),
                input_schema: tool.input_schema.clone(),
                required_permission: PermissionMode::DangerFullAccess,
            })
            .collect()
    }

    fn find_tool(&self, name: &str) -> Option<&McpToolBinding> {
        self.tools.iter().find(|tool| tool.exposed_name == name)
    }
}

fn run_repl(
    model: String,
    allowed_tools: Option<AllowedToolSet>,
    permission_mode: PermissionMode,
    collaboration_mode: CollaborationMode,
    reasoning_effort: Option<ReasoningEffort>,
    fast_mode: FastMode,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut cli = LiveCli::new(
        model,
        true,
        allowed_tools,
        permission_mode,
        collaboration_mode,
        reasoning_effort,
        fast_mode,
        true,
    )?;
    run_repl_loop(&mut cli)
}

fn run_repl_from_session(
    handle: SessionHandle,
    session: Session,
) -> Result<(), Box<dyn std::error::Error>> {
    let model = session
        .metadata
        .as_ref()
        .map(|metadata| metadata.model.clone())
        .unwrap_or_else(|| default_model_or(DEFAULT_MODEL));
    let mut cli = LiveCli::from_session(
        handle,
        session,
        model,
        None,
        default_permission_mode(),
        CollaborationMode::Build,
        None,
        FastMode::Off,
        true,
    )?;
    run_repl_loop(&mut cli)
}

fn run_repl_loop(cli: &mut LiveCli) -> Result<(), Box<dyn std::error::Error>> {
    let mut editor = input::LineEditor::new(ui::prompt_string(), repl_completion_candidates(cli));

    // Welcome banner: a single bordered panel that tells the user who they
    // are talking to, how the agent is configured, and which keystrokes to
    // remember. Printed once per REPL session.
    let cwd_display = env::current_dir()
        .ok()
        .map(|path| path.display().to_string());
    let provider_label = provider_label_for_service_model(cli.service, &cli.model);
    let banner = ui::welcome_banner(&ui::BannerInfo {
        version: VERSION,
        service: cli.service.display_name(),
        model: &cli.model,
        provider: provider_label.as_deref(),
        permission_mode: cli.permission_mode.as_str(),
        cwd: cwd_display.as_deref(),
    });
    println!("{banner}");
    println!();

    loop {
        editor.set_completions(repl_completion_candidates(cli));
        editor.set_status_line(Some(cli.prompt_status_line()));
        let input = match editor.read_line()? {
            input::ReadOutcome::Submit(input) => input,
            input::ReadOutcome::Cancel => continue,
            input::ReadOutcome::Exit => break,
            input::ReadOutcome::ToggleMode => {
                cli.toggle_mode()?;
                continue;
            }
        };
        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }
        editor.push_history(trimmed.to_string());
        if let Some(login_command) = parse_auth_command(trimmed) {
            login(login_command.service, login_command.api_key)?;
            continue;
        }
        if let Some(logout_command) = parse_logout_command(trimmed) {
            logout(logout_command.service)?;
            continue;
        }
        if let Some(model) = parse_model_command(trimmed) {
            match model {
                Some(model) => cli.set_model(model)?,
                None => {
                    if let Some(model) = open_model_picker()?.selected_model {
                        cli.set_model(model)?;
                    }
                }
            }
            continue;
        }
        if let Some(provider) = parse_provider_command(trimmed) {
            match provider {
                Some(provider) if is_clear_provider_arg(&provider) => cli.set_provider(None)?,
                Some(provider) => cli.set_provider(Some(provider))?,
                None => match open_provider_picker(&cli.model)?.selected_provider {
                    Some(provider) => cli.set_provider(Some(provider))?,
                    None => cli.set_provider(None)?,
                },
            }
            continue;
        }
        if let Some(mode) = parse_proxy_command(trimmed) {
            handle_proxy_runtime_command(cli, mode?)?;
            continue;
        }
        if let Some(reasoning_effort) = parse_reasoning_command(trimmed) {
            cli.set_reasoning(reasoning_effort?)?;
            continue;
        }
        if let Some(mode) = parse_mode_command(trimmed) {
            cli.set_mode(mode?)?;
            continue;
        }
        if let Some(fast_mode) = parse_fast_command(trimmed) {
            cli.set_fast_mode(fast_mode?)?;
            continue;
        }
        if let Some(command) = parse_mcp_command(trimmed) {
            handle_mcp_runtime_command(cli, command?)?;
            continue;
        }
        if let Some(mode) = parse_permissions_command(trimmed) {
            cli.set_permissions(mode?)?;
            continue;
        }
        match trimmed {
            "/exit" | "/quit" => break,
            _ if trimmed.starts_with('/') => {
                let Some(command) = SlashCommand::parse(trimmed) else {
                    continue;
                };
                match command {
                    SlashCommand::Help { topic } => println!(
                        "{}",
                        topic.as_deref().map_or_else(render_repl_help, |topic| {
                            render_slash_command_help_topic(Some(topic))
                        },)
                    ),
                    SlashCommand::Status => cli.print_status(),
                    SlashCommand::Compact => cli.compact()?,
                    SlashCommand::Reasoning { effort } => {
                        cli.set_reasoning(match effort.as_deref() {
                            Some(value) => Some(parse_reasoning_effort_arg(value)?),
                            None => None,
                        })?
                    }
                    SlashCommand::Fast { enabled } => {
                        cli.set_fast_mode(enabled.map(|enabled| {
                            if enabled {
                                FastMode::On
                            } else {
                                FastMode::Off
                            }
                        }))?
                    }
                    SlashCommand::Mode { mode } => cli.set_mode(
                        mode.as_deref()
                            .map(parse_collaboration_mode_arg)
                            .transpose()?,
                    )?,
                    SlashCommand::Permissions { mode } => cli.set_permissions(
                        mode.as_deref().map(parse_permission_mode_arg).transpose()?,
                    )?,
                    SlashCommand::Clear { confirm } => cli.clear_session(confirm)?,
                    SlashCommand::Resume { session_path } => cli.resume_session(session_path)?,
                    SlashCommand::Config { section } => {
                        println!("{}", render_config_report(section.as_deref())?)
                    }
                    SlashCommand::Memory => println!("{}", render_memory_report()?),
                    SlashCommand::Init => run_init_with_model(cli.service, &cli.model)?,
                    SlashCommand::Diff => println!("{}", render_diff_report()?),
                    SlashCommand::Version => print_version(),
                    SlashCommand::Branch { action, target } => println!(
                        "{}",
                        handle_branch_slash_command(
                            action.as_deref(),
                            target.as_deref(),
                            &env::current_dir()?,
                        )?
                    ),
                    SlashCommand::Worktree {
                        action,
                        path,
                        branch,
                    } => println!(
                        "{}",
                        handle_worktree_slash_command(
                            action.as_deref(),
                            path.as_deref(),
                            branch.as_deref(),
                            &env::current_dir()?,
                        )?
                    ),
                    SlashCommand::Export { path } => cli.export_session(path.as_deref())?,
                    SlashCommand::Session { action, target } => {
                        cli.handle_session_command(action.as_deref(), target.as_deref())?
                    }
                    SlashCommand::Sessions => println!("{}", render_session_list(&cli.session.id)?),
                    SlashCommand::Plugins { action, target } => {
                        cli.handle_plugins_command(action.as_deref(), target.as_deref())?
                    }
                    SlashCommand::Agents { args } => println!(
                        "{}",
                        handle_agents_slash_command(args.as_deref(), &env::current_dir()?)?
                    ),
                    SlashCommand::Skills { args } => println!(
                        "{}",
                        handle_skills_slash_command(args.as_deref(), &env::current_dir()?)?
                    ),
                    SlashCommand::Unknown(name) => eprintln!("unknown slash command: /{name}"),
                    SlashCommand::Model { .. }
                    | SlashCommand::Logout { .. }
                    | SlashCommand::Mcp { .. } => {
                        unreachable!("handled before shared slash command dispatch")
                    }
                }
            }
            _ => cli.run_turn(trimmed)?,
        }
    }

    Ok(())
}

struct LiveCli {
    service: ApiService,
    model: String,
    allowed_tools: Option<AllowedToolSet>,
    permission_mode: PermissionMode,
    collaboration_mode: CollaborationMode,
    reasoning_effort: Option<ReasoningEffort>,
    fast_mode: FastMode,
    system_prompt: Vec<String>,
    proxy_tool_calls: bool,
    mcp_catalog: McpCatalog,
    runtime: ConversationRuntime<PebbleRuntimeClient, CliToolExecutor>,
    session: SessionHandle,
    render_model_output: bool,
}

impl LiveCli {
    fn new(
        model: String,
        enable_tools: bool,
        allowed_tools: Option<AllowedToolSet>,
        permission_mode: PermissionMode,
        collaboration_mode: CollaborationMode,
        reasoning_effort: Option<ReasoningEffort>,
        fast_mode: FastMode,
        render_model_output: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let service = infer_service_for_model(&model);
        let system_prompt = build_system_prompt(service, &model, collaboration_mode)?;
        let proxy_tool_calls = proxy_tool_calls_enabled();
        let mcp_catalog = load_mcp_catalog(&env::current_dir()?)?;
        let session = create_managed_session_handle()?;
        auto_compact_inactive_sessions(&session.id)?;
        let runtime = build_runtime(
            Session::new(),
            service,
            model.clone(),
            system_prompt.clone(),
            enable_tools,
            proxy_tool_calls,
            mcp_catalog.clone(),
            allowed_tools.clone(),
            permission_mode,
            collaboration_mode,
            reasoning_effort,
            fast_mode,
            render_model_output,
        )?;
        let cli = Self {
            service,
            model,
            allowed_tools,
            permission_mode,
            collaboration_mode,
            reasoning_effort,
            fast_mode,
            system_prompt,
            proxy_tool_calls,
            mcp_catalog,
            runtime,
            session,
            render_model_output,
        };
        cli.persist_session()?;
        Ok(cli)
    }

    fn from_session(
        session_handle: SessionHandle,
        session: Session,
        model: String,
        allowed_tools: Option<AllowedToolSet>,
        permission_mode: PermissionMode,
        collaboration_mode: CollaborationMode,
        reasoning_effort: Option<ReasoningEffort>,
        fast_mode: FastMode,
        render_model_output: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let restored = session_runtime_state(
            &session,
            &model,
            allowed_tools.as_ref(),
            permission_mode,
            collaboration_mode,
            reasoning_effort,
            fast_mode,
            proxy_tool_calls_enabled(),
        );
        let system_prompt = build_system_prompt(
            restored.service,
            &restored.model,
            restored.collaboration_mode,
        )?;
        let mcp_catalog = load_mcp_catalog(&env::current_dir()?)?;
        let runtime = build_runtime(
            session,
            restored.service,
            restored.model.clone(),
            system_prompt.clone(),
            true,
            restored.proxy_tool_calls,
            mcp_catalog.clone(),
            restored.allowed_tools.clone(),
            restored.permission_mode,
            restored.collaboration_mode,
            restored.reasoning_effort,
            restored.fast_mode,
            render_model_output,
        )?;
        Ok(Self {
            service: restored.service,
            model: restored.model,
            allowed_tools: restored.allowed_tools,
            permission_mode: restored.permission_mode,
            collaboration_mode: restored.collaboration_mode,
            reasoning_effort: restored.reasoning_effort,
            fast_mode: restored.fast_mode,
            system_prompt,
            proxy_tool_calls: restored.proxy_tool_calls,
            mcp_catalog,
            runtime,
            session: session_handle,
            render_model_output,
        })
    }

    fn run_turn_with_output(
        &mut self,
        input: &str,
        output_format: CliOutputFormat,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match output_format {
            CliOutputFormat::Text => self.run_turn(input),
            CliOutputFormat::Json => self.run_prompt_json(input),
        }
    }

    fn run_turn(&mut self, input: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut spinner = Spinner::new();
        let mut stdout = io::stdout();
        // Visual turn separator so the transcript has clear "acts": one per
        // user message. Muted colour so it never steals attention from the
        // actual content above or below.
        println!("{}", ui::turn_separator());
        spinner.tick(
            if self.collaboration_mode == CollaborationMode::Plan {
                "planning"
            } else {
                "thinking"
            },
            TerminalRenderer::new().color_theme(),
            &mut stdout,
        )?;
        let mut permission_prompter = CliPermissionPrompter::new(self.permission_mode);
        let result = self.runtime.run_turn(input, Some(&mut permission_prompter));
        match result {
            Ok(summary) => {
                spinner.finish("done", TerminalRenderer::new().color_theme(), &mut stdout)?;
                self.persist_session()?;
                if let Some(event) = summary.auto_compaction {
                    println!(
                        "{}",
                        ui::dim_note(&format_auto_compaction_notice(event.removed_message_count))
                    );
                }
                println!();
                Ok(())
            }
            Err(error) => {
                spinner.fail(
                    "request failed",
                    TerminalRenderer::new().color_theme(),
                    &mut stdout,
                )?;
                Err(Box::new(error))
            }
        }
    }

    fn run_prompt_json(&mut self, input: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut permission_prompter = CliPermissionPrompter::new(self.permission_mode);
        let summary = self
            .runtime
            .run_turn(input, Some(&mut permission_prompter))?;
        self.persist_session()?;
        println!(
            "{}",
            serde_json::json!({
                "message": assistant_text_from_messages(&summary.assistant_messages),
                "model": self.model,
                "iterations": summary.iterations,
                "auto_compaction": summary.auto_compaction.map(|event| serde_json::json!({
                    "removed_messages": event.removed_message_count,
                    "notice": format_auto_compaction_notice(event.removed_message_count),
                })),
                "tool_uses": collect_tool_uses(&summary),
                "tool_results": collect_tool_results(&summary),
                "usage": {
                    "input_tokens": summary.usage.input_tokens,
                    "output_tokens": summary.usage.output_tokens,
                    "cache_creation_input_tokens": summary.usage.cache_creation_input_tokens,
                    "cache_read_input_tokens": summary.usage.cache_read_input_tokens,
                }
            })
        );
        Ok(())
    }

    /// Build the one-line status strip rendered above the input prompt.
    ///
    /// The REPL calls this before every input read so the strip always
    /// reflects the agent's current configuration. We deliberately
    /// compute this fresh each time instead of caching so toggled settings
    /// (mode, reasoning, permissions, provider) take effect immediately.
    fn prompt_status_line(&self) -> String {
        let estimated = self.runtime.estimated_tokens();
        ui::prompt_status_line(&ui::PromptStatusInfo {
            model: short_model_name(&self.model),
            permission_mode: self.permission_mode.as_str(),
            collaboration_mode: self.collaboration_mode.as_str(),
            reasoning_effort: reasoning_effort_label(self.effective_reasoning_effort()),
            fast_mode: self.fast_mode.enabled(),
            proxy_tool_calls: self.proxy_tool_calls,
            estimated_tokens: (estimated > 0)
                .then_some(u64::try_from(estimated).unwrap_or(u64::MAX)),
        })
    }

    fn print_status(&self) {
        let cumulative = self.runtime.usage().cumulative_usage();
        let latest = self.runtime.usage().current_turn_usage();
        let context = status_context(Some(&self.session.path)).expect("status context should load");
        println!(
            "{}",
            format_status_report(
                self.service,
                &self.model,
                StatusUsage {
                    message_count: self.runtime.session().messages.len(),
                    turns: self.runtime.usage().turns(),
                    latest,
                    cumulative,
                    estimated_tokens: self.runtime.estimated_tokens(),
                },
                self.permission_mode.as_str(),
                provider_label_for_service_model(self.service, &self.model).as_deref(),
                self.proxy_tool_calls,
                self.collaboration_mode,
                self.effective_reasoning_effort(),
                self.fast_mode,
                &self.mcp_catalog,
                &context,
            )
        );
        if self.model == DEFAULT_MODEL {
            println!();
            println!("  Defaults");
            println!("    fallback_model   active");
        } else {
            println!();
            println!("  Defaults");
            println!("    fallback_model   {DEFAULT_MODEL}");
        }
    }

    fn compact(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let result = self.runtime.compact(CompactionConfig::default());
        let removed = result.removed_message_count;
        self.rebuild_runtime(result.compacted_session)?;
        self.persist_session()?;
        println!("Compacted {removed} messages.");
        Ok(())
    }

    fn set_model(&mut self, model: String) -> Result<(), Box<dyn std::error::Error>> {
        let model = resolve_model_alias(&model).to_string();
        persist_current_model(model.clone())?;
        self.service = infer_service_for_model(&model);
        self.model = model.clone();
        self.rebuild_runtime(self.runtime.session().clone())?;
        self.persist_session()?;
        println!("Switched to service: {}", self.service.display_name());
        println!("Switched to model: {model}");
        if let Some(provider) = provider_label_for_service_model(self.service, &self.model) {
            println!("Provider override for current model: {provider}");
        }
        Ok(())
    }

    fn set_provider(&mut self, provider: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
        if self.service != ApiService::NanoGpt {
            return Err(format!(
                "provider overrides are only supported for NanoGPT models; current model {} is on {}",
                self.model,
                self.service.display_name()
            )
            .into());
        }
        let session = self.runtime.session().clone();
        if let Some(provider) = provider.as_deref() {
            validate_provider_for_model(&self.model, provider)?;
        }
        let provider_label = provider
            .clone()
            .unwrap_or_else(|| "<platform default>".to_string());
        persist_provider_for_model(&self.model, provider)?;
        self.rebuild_runtime(session)?;
        self.persist_session()?;
        if provider_label == "<platform default>" {
            println!(
                "Provider override for current model {}: {}",
                self.model, provider_label
            );
        } else {
            println!(
                "Provider override for current model {}: {} (paygo routing enabled)",
                self.model, provider_label
            );
        }
        Ok(())
    }

    fn set_proxy_tool_calls(&mut self, enabled: bool) -> Result<(), Box<dyn std::error::Error>> {
        let session = self.runtime.session().clone();
        persist_proxy_tool_calls(enabled)?;
        self.proxy_tool_calls = enabled;
        self.rebuild_runtime(session)?;
        self.persist_session()?;
        println!(
            "Proxy tool-call translation: {}",
            if enabled { "enabled" } else { "disabled" }
        );
        if enabled {
            println!(
                "Native tool schemas are disabled; tool use now goes through XML <tool_call> blocks."
            );
        }
        Ok(())
    }

    fn reload_mcp(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let session = self.runtime.session().clone();
        self.mcp_catalog = load_mcp_catalog(&env::current_dir()?)?;
        self.rebuild_runtime(session)?;
        self.persist_session()?;
        println!("Reloaded MCP config.");
        self.print_mcp_status();
        Ok(())
    }

    fn print_mcp_status(&self) {
        print_mcp_status(&self.mcp_catalog);
    }

    fn print_mcp_tools(&self) {
        print_mcp_tools(&self.mcp_catalog);
    }

    fn set_permissions(
        &mut self,
        mode: Option<PermissionMode>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(mode) = mode else {
            println!("Permission mode: {}", self.permission_mode.as_str());
            return Ok(());
        };
        if mode == self.permission_mode {
            println!("Permission mode: {}", self.permission_mode.as_str());
            return Ok(());
        }

        self.permission_mode = mode;
        self.rebuild_runtime(self.runtime.session().clone())?;
        self.persist_session()?;
        println!("Permission mode: {}", self.permission_mode.as_str());
        Ok(())
    }

    fn toggle_mode(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.set_mode(Some(match self.collaboration_mode {
            CollaborationMode::Build => CollaborationMode::Plan,
            CollaborationMode::Plan => CollaborationMode::Build,
        }))
    }

    fn set_mode(
        &mut self,
        mode: Option<CollaborationMode>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(mode) = mode else {
            println!("{}", format_mode_report(self.collaboration_mode));
            return Ok(());
        };
        if mode == self.collaboration_mode {
            println!("{}", format_mode_report(self.collaboration_mode));
            return Ok(());
        }

        self.collaboration_mode = mode;
        self.rebuild_runtime(self.runtime.session().clone())?;
        self.persist_session()?;
        println!("{}", format_mode_switch_report(self.collaboration_mode));
        Ok(())
    }

    fn set_reasoning(
        &mut self,
        reasoning_effort: Option<Option<ReasoningEffort>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(reasoning_effort) = reasoning_effort else {
            println!(
                "{}",
                format_reasoning_report(self.reasoning_effort, self.effective_reasoning_effort())
            );
            return Ok(());
        };
        if reasoning_effort == self.reasoning_effort {
            println!(
                "{}",
                format_reasoning_report(self.reasoning_effort, self.effective_reasoning_effort())
            );
            return Ok(());
        }

        self.reasoning_effort = reasoning_effort;
        self.rebuild_runtime(self.runtime.session().clone())?;
        self.persist_session()?;
        println!(
            "{}",
            format_reasoning_switch_report(
                self.reasoning_effort,
                self.effective_reasoning_effort()
            )
        );
        Ok(())
    }

    fn set_fast_mode(
        &mut self,
        fast_mode: Option<FastMode>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(fast_mode) = fast_mode else {
            println!("{}", format_fast_mode_report(self.fast_mode));
            return Ok(());
        };
        if fast_mode == self.fast_mode {
            println!("{}", format_fast_mode_report(self.fast_mode));
            return Ok(());
        }

        self.fast_mode = fast_mode;
        self.rebuild_runtime(self.runtime.session().clone())?;
        self.persist_session()?;
        println!("{}", format_fast_mode_switch_report(self.fast_mode));
        Ok(())
    }

    fn effective_reasoning_effort(&self) -> Option<ReasoningEffort> {
        effective_reasoning_effort(self.collaboration_mode, self.reasoning_effort)
    }

    fn rebuild_runtime(&mut self, session: Session) -> Result<(), Box<dyn std::error::Error>> {
        self.system_prompt =
            build_system_prompt(self.service, &self.model, self.collaboration_mode)?;
        self.runtime = build_runtime(
            session,
            self.service,
            self.model.clone(),
            self.system_prompt.clone(),
            true,
            self.proxy_tool_calls,
            self.mcp_catalog.clone(),
            self.allowed_tools.clone(),
            self.permission_mode,
            self.collaboration_mode,
            self.reasoning_effort,
            self.fast_mode,
            self.render_model_output,
        )?;
        Ok(())
    }

    fn handle_plugins_command(
        &mut self,
        action: Option<&str>,
        target: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("{}", handle_plugins_command(action, target)?);
        if plugins_command_is_mutating(action) {
            self.rebuild_runtime(self.runtime.session().clone())?;
            self.persist_session()?;
        }
        Ok(())
    }

    fn clear_session(&mut self, confirm: bool) -> Result<(), Box<dyn std::error::Error>> {
        if !confirm {
            println!(
                "clear: confirmation required; run /clear --confirm to start a fresh session."
            );
            return Ok(());
        }

        self.session = create_managed_session_handle()?;
        self.rebuild_runtime(Session::new())?;
        self.persist_session()?;
        println!(
            "Session cleared\n  Mode             fresh session\n  Preserved model  {}\n  Session mode     {}\n  Permission mode  {}\n  Session          {}",
            self.model,
            self.collaboration_mode.as_str(),
            self.permission_mode.as_str(),
            self.session.id,
        );
        Ok(())
    }

    fn resume_session(
        &mut self,
        session_path: Option<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let session_ref = match session_path {
            Some(session_ref) => session_ref,
            None => match prompt_for_session_selection(Some(&self.session.id))? {
                Some(handle) => {
                    return self.resume_handle(handle);
                }
                None => return Ok(()),
            },
        };
        let handle = resolve_session_reference(&session_ref)?;
        self.resume_handle(handle)
    }

    fn resume_handle(&mut self, handle: SessionHandle) -> Result<(), Box<dyn std::error::Error>> {
        let session = Session::load_from_path(&handle.path)?;
        let message_count = session.messages.len();
        self.restore_session_runtime(handle.clone(), session)?;
        self.session = handle;
        self.persist_session()?;
        println!(
            "{}",
            format_resume_report(
                &self.session.path.display().to_string(),
                message_count,
                self.runtime.usage().turns(),
            )
        );
        Ok(())
    }

    fn export_session(
        &self,
        requested_path: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let export_path = resolve_export_path(requested_path, self.runtime.session())?;
        fs::write(&export_path, render_export_text(self.runtime.session()))?;
        println!(
            "Export\n  Result           wrote transcript\n  File             {}\n  Messages         {}",
            export_path.display(),
            self.runtime.session().messages.len(),
        );
        Ok(())
    }

    fn handle_session_command(
        &mut self,
        action: Option<&str>,
        target: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match action {
            None | Some("list") => println!("{}", render_session_list(&self.session.id)?),
            Some("switch") => {
                let Some(target) = target else {
                    println!("Usage: /session switch <session-id>");
                    return Ok(());
                };
                let handle = resolve_session_reference(target)?;
                let session = Session::load_from_path(&handle.path)?;
                let message_count = session.messages.len();
                self.restore_session_runtime(handle.clone(), session)?;
                self.session = handle;
                self.persist_session()?;
                println!(
                    "Session switched\n  Active session   {}\n  File             {}\n  Messages         {}",
                    self.session.id,
                    self.session.path.display(),
                    message_count,
                );
            }
            Some(other) => println!(
                "Unknown /session action '{other}'. Use /session list or /session switch <session-id>."
            ),
        }
        Ok(())
    }

    fn restore_session_runtime(
        &mut self,
        handle: SessionHandle,
        session: Session,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let state = session_runtime_state(
            &session,
            &self.model,
            self.allowed_tools.as_ref(),
            self.permission_mode,
            self.collaboration_mode,
            self.reasoning_effort,
            self.fast_mode,
            self.proxy_tool_calls,
        );
        self.model = state.model;
        self.service = state.service;
        self.allowed_tools = state.allowed_tools;
        self.permission_mode = state.permission_mode;
        self.collaboration_mode = state.collaboration_mode;
        self.reasoning_effort = state.reasoning_effort;
        self.fast_mode = state.fast_mode;
        self.proxy_tool_calls = state.proxy_tool_calls;
        self.mcp_catalog = load_mcp_catalog(&env::current_dir()?)?;
        self.rebuild_runtime(session)?;
        self.session = handle;
        Ok(())
    }

    fn persist_session(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut session = self.runtime.session().clone();
        session.metadata = Some(derive_session_metadata(
            &session,
            &self.model,
            self.allowed_tools.as_ref(),
            self.permission_mode,
            self.collaboration_mode,
            self.reasoning_effort,
            self.fast_mode,
            self.proxy_tool_calls,
        ));
        session.save_to_path(&self.session.path)?;
        auto_compact_inactive_sessions(&self.session.id)?;
        Ok(())
    }
}

fn handle_proxy_runtime_command(
    cli: &mut LiveCli,
    mode: ProxyCommand,
) -> Result<(), Box<dyn std::error::Error>> {
    match mode {
        ProxyCommand::Status => {
            println!(
                "Proxy tool-call translation: {}",
                if cli.proxy_tool_calls {
                    "enabled"
                } else {
                    "disabled"
                }
            );
            Ok(())
        }
        ProxyCommand::Toggle => cli.set_proxy_tool_calls(!cli.proxy_tool_calls),
        ProxyCommand::Enable => cli.set_proxy_tool_calls(true),
        ProxyCommand::Disable => cli.set_proxy_tool_calls(false),
    }
}

fn handle_mcp_runtime_command(
    cli: &mut LiveCli,
    command: McpCommand,
) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        McpCommand::Status => {
            cli.print_mcp_status();
            Ok(())
        }
        McpCommand::Tools => {
            cli.print_mcp_tools();
            Ok(())
        }
        McpCommand::Reload => cli.reload_mcp(),
        McpCommand::Add { name } => {
            println!(
                "{}",
                add_mcp_server_interactive(&env::current_dir()?, &name)?
            );
            cli.reload_mcp()
        }
        McpCommand::Enable { name } => {
            println!(
                "{}",
                set_mcp_server_enabled(&env::current_dir()?, &name, true)?
            );
            cli.reload_mcp()
        }
        McpCommand::Disable { name } => {
            println!(
                "{}",
                set_mcp_server_enabled(&env::current_dir()?, &name, false)?
            );
            cli.reload_mcp()
        }
    }
}

struct CliPermissionPrompter {
    current_mode: PermissionMode,
}

impl CliPermissionPrompter {
    fn new(current_mode: PermissionMode) -> Self {
        Self { current_mode }
    }
}

impl PermissionPrompter for CliPermissionPrompter {
    fn decide(&mut self, request: &PermissionRequest) -> PermissionPromptDecision {
        println!();
        println!("Permission approval required");
        println!("  Tool             {}", request.tool_name);
        println!("  Current mode     {}", self.current_mode.as_str());
        println!("  Required mode    {}", request.required_mode.as_str());
        println!("  Input            {}", request.input);
        print!("Approve this tool call? [y/N]: ");
        let _ = io::stdout().flush();

        let mut response = String::new();
        match io::stdin().read_line(&mut response) {
            Ok(_) => {
                let normalized = response.trim().to_ascii_lowercase();
                if matches!(normalized.as_str(), "y" | "yes") {
                    PermissionPromptDecision::Allow
                } else {
                    PermissionPromptDecision::Deny {
                        reason: format!(
                            "tool '{}' denied by user approval prompt",
                            request.tool_name
                        ),
                    }
                }
            }
            Err(error) => PermissionPromptDecision::Deny {
                reason: format!("permission approval failed: {error}"),
            },
        }
    }
}

fn build_system_prompt(
    service: ApiService,
    model: &str,
    collaboration_mode: CollaborationMode,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut prompt = load_system_prompt_with_model_family(
        env::current_dir()?,
        DEFAULT_DATE,
        env::consts::OS,
        "unknown",
        prompt_model_family(service, model),
    )?;
    prompt.push(
        [
            "# Web Research Guidance",
            " - Use WebSearch when you need current or web-sourced information, such as docs discovery, release notes, changelogs, product details, or anything that may have changed recently.",
            " - Use WebScrape when you already know the URL and need to read the page contents closely, especially documentation pages, blog posts, articles, and reference material.",
            " - Use WebFetch only for a quick single-page fetch/summary when you do not need richer scraping output.",
            " - For documentation work, the preferred sequence is usually: WebSearch to find the right page, then WebScrape to read it carefully.",
        ]
        .join("\n"),
    );
    prompt.push(mode_instructions(collaboration_mode));
    Ok(prompt)
}

fn mode_instructions(collaboration_mode: CollaborationMode) -> String {
    match collaboration_mode {
        CollaborationMode::Build => [
            "# Collaboration Mode: Build",
            "",
            "You are in Build mode. Execute the user's request directly and make the requested changes when appropriate.",
            "Prefer taking action over only proposing plans, and make reasonable assumptions when the repo can answer them.",
        ]
        .join("\n"),
        CollaborationMode::Plan => [
            "# Collaboration Mode: Plan",
            "",
            "You are in Plan mode until the system prompt changes again.",
            "Plan mode is not changed by user intent or imperative phrasing; if the user asks you to execute, plan that execution instead of doing it.",
            "",
            "Rules:",
            " - You may explore the repo, inspect files, and run non-mutating commands that improve the plan.",
            " - You must not edit files, apply patches, run mutating formatters/codegen, or otherwise perform implementation work.",
            " - Ask focused questions only when the answer cannot be discovered from the environment and materially changes the plan.",
            " - Final answers in this mode should be implementation-ready plans, not code changes.",
        ]
        .join("\n"),
    }
}

fn prompt_model_family(service: ApiService, model: &str) -> String {
    match service {
        ApiService::NanoGpt => "NanoGPT Messages API".to_string(),
        ApiService::Synthetic => format!("Synthetic ({model})"),
        ApiService::OpenAiCodex => format!("OpenAI Codex ({model})"),
        ApiService::OpencodeGo => format!("OpenCode Go ({model})"),
    }
}

fn build_runtime_feature_config(
) -> Result<runtime::RuntimeFeatureConfig, Box<dyn std::error::Error>> {
    let cwd = env::current_dir()?;
    let loader = ConfigLoader::default_for(&cwd);
    let runtime_config = loader.load()?;
    let mut feature_config = runtime_config.feature_config().clone();
    let plugin_manager = build_plugin_manager(&cwd, &loader, &runtime_config);
    let plugin_hooks = plugin_manager.aggregated_hooks()?;
    let plugin_pre_hooks = plugin_hooks.pre_tool_use.clone();
    let plugin_post_hooks = plugin_hooks.post_tool_use.clone();
    let merged_hooks = runtime::RuntimeHookConfig::new(
        feature_config
            .hooks()
            .pre_tool_use()
            .iter()
            .cloned()
            .chain(plugin_pre_hooks)
            .collect(),
        feature_config
            .hooks()
            .post_tool_use()
            .iter()
            .cloned()
            .chain(plugin_post_hooks)
            .collect(),
        feature_config.hooks().post_tool_use_failure().to_vec(),
    );
    feature_config = feature_config.with_hooks(merged_hooks);
    Ok(feature_config)
}

fn build_runtime(
    session: Session,
    service: ApiService,
    model: String,
    system_prompt: Vec<String>,
    enable_tools: bool,
    proxy_tool_calls: bool,
    mcp_catalog: McpCatalog,
    allowed_tools: Option<AllowedToolSet>,
    permission_mode: PermissionMode,
    collaboration_mode: CollaborationMode,
    reasoning_effort: Option<ReasoningEffort>,
    fast_mode: FastMode,
    render_model_output: bool,
) -> Result<ConversationRuntime<PebbleRuntimeClient, CliToolExecutor>, Box<dyn std::error::Error>> {
    let tool_registry = current_tool_registry()
        .map_err(|error| io::Error::other(format!("failed to load tool registry: {error}")))?;
    let tool_specs = if enable_tools {
        filter_runtime_tool_specs(
            available_runtime_tool_specs(&tool_registry, &mcp_catalog),
            allowed_tools.as_ref(),
        )
    } else {
        Vec::new()
    };
    let mut runtime_prompt = system_prompt;
    if enable_tools && proxy_tool_calls {
        runtime_prompt.push(build_proxy_system_prompt(&tool_specs));
    }
    let permission_policy = permission_policy(
        permission_mode,
        &available_runtime_tool_specs(&tool_registry, &mcp_catalog),
    );
    let max_output_tokens = max_output_tokens_for_model_or(&model, DEFAULT_MAX_TOKENS);
    let auto_compaction_threshold = configured_auto_compaction_threshold().unwrap_or_else(|| {
        derive_auto_compaction_threshold(&model, max_output_tokens)
            .unwrap_or_else(auto_compaction_threshold_from_env)
    });
    Ok(ConversationRuntime::new_with_features(
        session,
        PebbleRuntimeClient::new(
            service,
            model.clone(),
            max_output_tokens,
            provider_for_model(&model),
            enable_tools,
            proxy_tool_calls,
            tool_specs.clone(),
            collaboration_mode,
            reasoning_effort,
            fast_mode,
            render_model_output,
        )?,
        CliToolExecutor::new(
            service,
            tool_registry,
            mcp_catalog,
            tool_specs,
            allowed_tools,
            render_model_output,
        ),
        permission_policy,
        runtime_prompt,
        build_runtime_feature_config()?,
    )
    .with_auto_compaction_input_tokens_threshold(auto_compaction_threshold))
}

fn configured_auto_compaction_threshold() -> Option<u32> {
    std::env::var("PEBBLE_AUTO_COMPACT_INPUT_TOKENS")
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .filter(|threshold| *threshold > 0)
}

fn derive_auto_compaction_threshold(model: &str, max_output_tokens: u32) -> Option<u32> {
    let context_length = context_length_for_model(model)?;
    let reserved_output_tokens =
        u64::from(max_output_tokens).saturating_add(AUTO_COMPACTION_CONTEXT_SAFETY_MARGIN_TOKENS);
    let available_input_tokens = context_length.saturating_sub(reserved_output_tokens);
    if available_input_tokens == 0 {
        return None;
    }

    let threshold =
        available_input_tokens.saturating_mul(AUTO_COMPACTION_CONTEXT_UTILIZATION_PERCENT) / 100;
    u32::try_from(threshold.max(1).min(u64::from(u32::MAX))).ok()
}

struct PebbleRuntimeClient {
    runtime: tokio::runtime::Runtime,
    service: ApiService,
    model: String,
    provider: Option<String>,
    max_output_tokens: u32,
    enable_tools: bool,
    proxy_tool_calls: bool,
    tool_specs: Vec<RuntimeToolSpec>,
    collaboration_mode: CollaborationMode,
    reasoning_effort: Option<ReasoningEffort>,
    fast_mode: FastMode,
    render_output: bool,
}

impl PebbleRuntimeClient {
    fn new(
        service: ApiService,
        model: String,
        max_output_tokens: u32,
        provider: Option<String>,
        enable_tools: bool,
        proxy_tool_calls: bool,
        tool_specs: Vec<RuntimeToolSpec>,
        collaboration_mode: CollaborationMode,
        reasoning_effort: Option<ReasoningEffort>,
        fast_mode: FastMode,
        render_output: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            runtime: tokio::runtime::Runtime::new()?,
            service,
            model,
            provider,
            max_output_tokens,
            enable_tools,
            proxy_tool_calls,
            tool_specs,
            collaboration_mode,
            reasoning_effort,
            fast_mode,
            render_output,
        })
    }
}

impl ApiClient for PebbleRuntimeClient {
    fn stream(&mut self, request: ApiRequest) -> Result<Vec<AssistantEvent>, RuntimeError> {
        if self.proxy_tool_calls {
            return self.stream_via_proxy(request);
        }

        let effective_reasoning_effort =
            effective_reasoning_effort(self.collaboration_mode, self.reasoning_effort);
        let message_request = MessageRequest {
            model: self.model.clone(),
            max_tokens: self.max_output_tokens,
            messages: convert_messages(&request.messages, self.service)?,
            system: (!request.system_prompt.is_empty()).then(|| request.system_prompt.join("\n\n")),
            tools: self.enable_tools.then(|| {
                self.tool_specs
                    .iter()
                    .map(|spec| ToolDefinition {
                        name: spec.name.clone(),
                        description: Some(spec.description.clone()),
                        input_schema: spec.input_schema.clone(),
                    })
                    .collect()
            }),
            tool_choice: self.enable_tools.then_some(ToolChoice::Auto),
            thinking: (self.service != ApiService::OpenAiCodex
                && effective_reasoning_effort.is_some())
            .then_some(ThinkingConfig::enabled(DEFAULT_THINKING_BUDGET_TOKENS)),
            reasoning_effort: effective_reasoning_effort,
            fast_mode: self.fast_mode.enabled(),
            stream: true,
        };

        let client = self.service_client()?;
        self.runtime.block_on(async {
            let mut stream = client
                .stream_message(&message_request)
                .await
                .map_err(|error| RuntimeError::new(error.to_string()))?;
            let mut output: Box<dyn Write> = if self.render_output {
                Box::new(io::stdout())
            } else {
                Box::new(io::sink())
            };
            let renderer = TerminalRenderer::new();
            let mut markdown_stream = MarkdownStreamState::default();
            let mut events = Vec::new();
            let mut pending_tool: Option<(String, String, String)> = None;
            let mut saw_stop = false;
            let mut stream_fallback_requested = false;
            // Track whether we've already printed the "thinking" lead so we
            // don't repeat it for each streamed thinking delta. The flag is
            // reset every time the outer loop re-enters this closure.
            let mut thinking_stream_started = false;
            // Print the "● pebble" assistant lead exactly once per streamed
            // response — right before the first text delta — so the model's
            // reply is visually anchored even after a wall of tool output.
            let mut assistant_lead_emitted = false;
            let thinking_enabled = effective_reasoning_effort.is_some();
            let render_output_enabled = self.render_output;

            loop {
                let event = match stream.next_event().await {
                    Ok(Some(event)) => event,
                    Ok(None) => break,
                    Err(ApiError::StreamApi {
                        error_type: Some(error_type),
                        message,
                        ..
                    }) if error_type == "invalid_response_error" => {
                        eprintln!(
                            "[pebble] streaming failed with invalid_response_error{}; retrying non-streaming",
                            message
                                .as_deref()
                                .map(|message| format!(": {message}"))
                                .unwrap_or_default()
                        );
                        stream_fallback_requested = true;
                        break;
                    }
                    Err(error) => return Err(RuntimeError::new(error.to_string())),
                };

                match event {
                    ApiStreamEvent::MessageStart(start) => {
                        for block in start.message.content {
                            push_output_block(
                                block,
                                output.as_mut(),
                                &mut events,
                                &mut pending_tool,
                                true,
                            )?;
                        }
                    }
                    ApiStreamEvent::ContentBlockStart(start) => {
                        push_output_block(
                            start.content_block,
                            output.as_mut(),
                            &mut events,
                            &mut pending_tool,
                            true,
                        )?;
                    }
                    ApiStreamEvent::ContentBlockDelta(delta) => match delta.delta {
                        ContentBlockDelta::TextDelta { text } => {
                            if !text.is_empty() {
                                if render_output_enabled && !assistant_lead_emitted {
                                    write!(output, "{}", ui::assistant_lead())
                                        .and_then(|_| output.flush())
                                        .map_err(|error| RuntimeError::new(error.to_string()))?;
                                    assistant_lead_emitted = true;
                                }
                                if let Some(rendered) = markdown_stream.push(&renderer, &text) {
                                    write!(output, "{rendered}")
                                        .and_then(|_| output.flush())
                                        .map_err(|error| RuntimeError::new(error.to_string()))?;
                                }
                                events.push(AssistantEvent::TextDelta(text));
                            }
                        }
                        ContentBlockDelta::ThinkingDelta { thinking } => {
                            if !thinking.is_empty() {
                                // Surface reasoning live when the user has
                                // opted in via `/thinking on`. We dim the
                                // text heavily so it reads as subordinate
                                // context rather than primary output.
                                if thinking_enabled && render_output_enabled {
                                    if !thinking_stream_started {
                                        write!(output, "{}", ui::thinking_lead())
                                            .and_then(|_| output.flush())
                                            .map_err(|error| {
                                                RuntimeError::new(error.to_string())
                                            })?;
                                        thinking_stream_started = true;
                                    }
                                    write!(output, "{}", ui::thinking_chunk(&thinking))
                                        .and_then(|_| output.flush())
                                        .map_err(|error| RuntimeError::new(error.to_string()))?;
                                }
                                events.push(AssistantEvent::ThinkingDelta(thinking));
                            }
                        }
                        ContentBlockDelta::SignatureDelta { signature } => {
                            if !signature.is_empty() {
                                events.push(AssistantEvent::ThinkingSignature(signature));
                            }
                        }
                        ContentBlockDelta::InputJsonDelta { partial_json } => {
                            if let Some((_, _, input)) = &mut pending_tool {
                                input.push_str(&partial_json);
                            }
                        }
                    },
                    ApiStreamEvent::ContentBlockStop(_) => {
                        if let Some(rendered) = markdown_stream.flush(&renderer) {
                            write!(output, "{rendered}")
                                .and_then(|_| output.flush())
                                .map_err(|error| RuntimeError::new(error.to_string()))?;
                        }
                        // If we were streaming a reasoning block, close it
                        // with a blank line so the subsequent assistant text
                        // visually separates from the thinking trail.
                        if thinking_stream_started {
                            writeln!(output)
                                .and_then(|_| output.flush())
                                .map_err(|error| RuntimeError::new(error.to_string()))?;
                            thinking_stream_started = false;
                        }
                        if let Some((id, name, input)) = pending_tool.take() {
                            render_streamed_tool_call_start(output.as_mut(), &name, &input)?;
                            events.push(AssistantEvent::ToolUse { id, name, input });
                        }
                    }
                    ApiStreamEvent::MessageDelta(delta) => {
                        events.push(AssistantEvent::Usage(TokenUsage {
                            input_tokens: delta.usage.input_tokens,
                            output_tokens: delta.usage.output_tokens,
                            cache_creation_input_tokens: 0,
                            cache_read_input_tokens: 0,
                        }));
                    }
                    ApiStreamEvent::MessageStop(_) => {
                        saw_stop = true;
                        if let Some(rendered) = markdown_stream.flush(&renderer) {
                            write!(output, "{rendered}")
                                .and_then(|_| output.flush())
                                .map_err(|error| RuntimeError::new(error.to_string()))?;
                        }
                        events.push(AssistantEvent::MessageStop);
                    }
                }
            }

            if !stream_fallback_requested
                && !saw_stop
                && events.iter().any(|event| {
                    matches!(event, AssistantEvent::TextDelta(text) if !text.is_empty())
                        || matches!(event, AssistantEvent::ThinkingDelta(text) if !text.is_empty())
                        || matches!(event, AssistantEvent::ThinkingSignature(_))
                        || matches!(event, AssistantEvent::ToolUse { .. })
                })
            {
                events.push(AssistantEvent::MessageStop);
            }

            if events
                .iter()
                .any(|event| matches!(event, AssistantEvent::MessageStop))
            {
                return Ok(events);
            }

            let response = client
                .send_message(&MessageRequest {
                    stream: false,
                    ..message_request.clone()
                })
                .await
                .map_err(|error| RuntimeError::new(error.to_string()))?;
            response_to_events(response, output.as_mut())
        })
    }
}

impl PebbleRuntimeClient {
    fn service_client(&self) -> Result<NanoGptClient, RuntimeError> {
        let mut client = NanoGptClient::from_service_env(self.service)
            .map_err(|error| RuntimeError::new(error.to_string()))?;
        if self.service == ApiService::NanoGpt {
            client = client.with_provider(self.provider.clone());
        }
        Ok(client)
    }

    fn stream_via_proxy(
        &mut self,
        request: ApiRequest,
    ) -> Result<Vec<AssistantEvent>, RuntimeError> {
        let effective_reasoning_effort =
            effective_reasoning_effort(self.collaboration_mode, self.reasoning_effort);
        let mut messages = convert_proxy_messages_to_input_messages(
            convert_messages_for_proxy(&request.messages).map_err(RuntimeError::new)?,
        );
        let base_message_request = MessageRequest {
            model: self.model.clone(),
            max_tokens: self.max_output_tokens,
            messages: messages.clone(),
            system: (!request.system_prompt.is_empty()).then(|| request.system_prompt.join("\n\n")),
            tools: None,
            tool_choice: None,
            thinking: (self.service != ApiService::OpenAiCodex
                && effective_reasoning_effort.is_some())
            .then_some(ThinkingConfig::enabled(DEFAULT_THINKING_BUDGET_TOKENS)),
            reasoning_effort: effective_reasoning_effort,
            fast_mode: self.fast_mode.enabled(),
            stream: false,
        };

        let client = self.service_client()?;
        self.runtime.block_on(async {
            let response = client
                .send_message(&base_message_request)
                .await
                .map_err(|error| RuntimeError::new(error.to_string()))?;
            let mut first_render = Vec::new();
            let first_events =
                proxy_response_to_events(response, &mut first_render, &self.tool_specs)?;

            if should_retry_proxy_tool_prompt(&first_events) {
                messages.push(InputMessage::user_text(proxy_retry_reminder()));
                let retry_request = MessageRequest {
                    messages,
                    ..base_message_request
                };
                let retry_response = client
                    .send_message(&retry_request)
                    .await
                    .map_err(|error| RuntimeError::new(error.to_string()))?;
                let mut output: Box<dyn Write> = if self.render_output {
                    Box::new(io::stdout())
                } else {
                    Box::new(io::sink())
                };
                return proxy_response_to_events(retry_response, output.as_mut(), &self.tool_specs);
            }

            let mut output: Box<dyn Write> = if self.render_output {
                Box::new(io::stdout())
            } else {
                Box::new(io::sink())
            };
            output
                .write_all(&first_render)
                .and_then(|_| output.flush())
                .map_err(|error| RuntimeError::new(error.to_string()))?;
            Ok(first_events)
        })
    }
}

fn proxy_retry_reminder() -> &'static str {
    "Your previous reply only narrated intent. Do not narrate. If tool use is needed, emit the next XML <tool_call> block immediately with no prefatory text. If no tool is needed, answer directly."
}

fn should_retry_proxy_tool_prompt(events: &[AssistantEvent]) -> bool {
    if events
        .iter()
        .any(|event| matches!(event, AssistantEvent::ToolUse { .. }))
    {
        return false;
    }

    let text = events
        .iter()
        .filter_map(|event| match event {
            AssistantEvent::TextDelta(text) => Some(text.as_str()),
            _ => None,
        })
        .collect::<String>();
    let trimmed = text.trim();
    if trimmed.is_empty() || trimmed.chars().count() > 280 {
        return false;
    }

    let normalized = trimmed.to_ascii_lowercase();
    let intent_prefix = [
        "let me ",
        "i'll ",
        "i will ",
        "first, i'll",
        "first i ",
        "let's ",
    ];
    let tool_intent = [
        "explore",
        "inspect",
        "look at",
        "check",
        "review",
        "understand",
        "start by",
        "begin by",
    ];

    intent_prefix
        .iter()
        .any(|prefix| normalized.starts_with(prefix))
        && tool_intent.iter().any(|phrase| normalized.contains(phrase))
}

fn push_output_block(
    block: OutputContentBlock,
    out: &mut (impl Write + ?Sized),
    events: &mut Vec<AssistantEvent>,
    pending_tool: &mut Option<(String, String, String)>,
    streaming_tool_input: bool,
) -> Result<(), RuntimeError> {
    match block {
        OutputContentBlock::Text { text } => {
            if !text.is_empty() {
                let rendered = TerminalRenderer::new().markdown_to_ansi(&text);
                write!(out, "{rendered}")
                    .and_then(|_| out.flush())
                    .map_err(|error| RuntimeError::new(error.to_string()))?;
                events.push(AssistantEvent::TextDelta(text));
            }
        }
        OutputContentBlock::Thinking {
            thinking,
            signature,
        } => {
            render_thinking_block_summary(&thinking, out)?;
            if !thinking.is_empty() {
                events.push(AssistantEvent::ThinkingDelta(thinking));
            }
            if let Some(signature) = signature.filter(|signature| !signature.is_empty()) {
                events.push(AssistantEvent::ThinkingSignature(signature));
            }
        }
        OutputContentBlock::RedactedThinking { .. } => {}
        OutputContentBlock::ToolUse { id, name, input } => {
            let initial_input = if streaming_tool_input
                && input.is_object()
                && input.as_object().is_some_and(|object| object.is_empty())
            {
                String::new()
            } else {
                input.to_string()
            };
            *pending_tool = Some((id, name, initial_input));
        }
    }
    Ok(())
}

fn render_streamed_tool_call_start(
    out: &mut (impl Write + ?Sized),
    name: &str,
    input: &str,
) -> Result<(), RuntimeError> {
    writeln!(out, "{}", format_tool_call_start(name, input))
        .and_then(|_| out.flush())
        .map_err(|error| RuntimeError::new(error.to_string()))
}

fn format_tool_call_start(name: &str, input: &str) -> String {
    let parsed =
        serde_json::from_str::<JsonValue>(input).unwrap_or(JsonValue::String(input.to_string()));
    let detail = match name {
        "bash" | "Bash" => parsed
            .get("command")
            .and_then(JsonValue::as_str)
            .map(|command| truncate_for_summary(command, 120))
            .unwrap_or_default(),
        "read_file" | "Read" => parsed
            .get("file_path")
            .or_else(|| parsed.get("path"))
            .and_then(JsonValue::as_str)
            .unwrap_or("?")
            .to_string(),
        "write_file" | "Write" => {
            let path = parsed
                .get("file_path")
                .or_else(|| parsed.get("path"))
                .and_then(JsonValue::as_str)
                .unwrap_or("?");
            let lines = parsed
                .get("content")
                .and_then(JsonValue::as_str)
                .map(|content| content.lines().count())
                .unwrap_or(0);
            format!("{path} ({lines} lines)")
        }
        "edit_file" | "Edit" => parsed
            .get("file_path")
            .or_else(|| parsed.get("path"))
            .and_then(JsonValue::as_str)
            .unwrap_or("?")
            .to_string(),
        "glob_search" | "Glob" | "grep_search" | "Grep" => parsed
            .get("pattern")
            .and_then(JsonValue::as_str)
            .unwrap_or("?")
            .to_string(),
        "web_search" | "WebSearch" => parsed
            .get("query")
            .and_then(JsonValue::as_str)
            .unwrap_or("?")
            .to_string(),
        _ => summarize_tool_payload(input),
    };

    ui::tool_call_header(name, &detail)
}

fn render_thinking_block_summary(
    _text: &str,
    _out: &mut (impl Write + ?Sized),
) -> Result<(), RuntimeError> {
    // Intentionally silent. Historically this printed a "reasoning hidden
    // (N chars)" note, but we already surface reasoning live via
    // `ui::thinking_chunk` when the user opts in to `/thinking on`, and
    // showing a stand-in summary otherwise is just noise — especially when
    // the same turn contains multiple thinking blocks, which caused the
    // note to repeat on every tool hop.
    Ok(())
}

fn response_to_events(
    response: MessageResponse,
    out: &mut (impl Write + ?Sized),
) -> Result<Vec<AssistantEvent>, RuntimeError> {
    let mut events = Vec::new();
    let mut pending_tool = None;

    for block in response.content {
        push_output_block(block, out, &mut events, &mut pending_tool, false)?;
        if let Some((id, name, input)) = pending_tool.take() {
            events.push(AssistantEvent::ToolUse { id, name, input });
        }
    }

    events.push(AssistantEvent::Usage(TokenUsage {
        input_tokens: response.usage.input_tokens,
        output_tokens: response.usage.output_tokens,
        cache_creation_input_tokens: response.usage.cache_creation_input_tokens,
        cache_read_input_tokens: response.usage.cache_read_input_tokens,
    }));
    events.push(AssistantEvent::MessageStop);
    Ok(events)
}

fn proxy_response_to_events(
    response: MessageResponse,
    out: &mut (impl Write + ?Sized),
    tool_specs: &[RuntimeToolSpec],
) -> Result<Vec<AssistantEvent>, RuntimeError> {
    let mut events = Vec::new();
    for block in response.content {
        match block {
            OutputContentBlock::Text { text } => {
                append_proxy_text_events(&text, out, &mut events, tool_specs)?;
            }
            OutputContentBlock::Thinking {
                thinking,
                signature,
            } => {
                render_thinking_block_summary(&thinking, out)?;
                if !thinking.is_empty() {
                    events.push(AssistantEvent::ThinkingDelta(thinking));
                }
                if let Some(signature) = signature.filter(|signature| !signature.is_empty()) {
                    events.push(AssistantEvent::ThinkingSignature(signature));
                }
            }
            OutputContentBlock::RedactedThinking { .. } => {}
            OutputContentBlock::ToolUse { id, name, input } => {
                events.push(AssistantEvent::ToolUse {
                    id,
                    name,
                    input: input.to_string(),
                });
            }
        }
    }

    events.push(AssistantEvent::Usage(TokenUsage {
        input_tokens: response.usage.input_tokens,
        output_tokens: response.usage.output_tokens,
        cache_creation_input_tokens: response.usage.cache_creation_input_tokens,
        cache_read_input_tokens: response.usage.cache_read_input_tokens,
    }));
    events.push(AssistantEvent::MessageStop);
    Ok(events)
}

fn append_proxy_text_events(
    text: &str,
    out: &mut (impl Write + ?Sized),
    events: &mut Vec<AssistantEvent>,
    tool_specs: &[RuntimeToolSpec],
) -> Result<(), RuntimeError> {
    let segments = parse_proxy_response(text, tool_specs).map_err(RuntimeError::new)?;
    let has_tool_use = segments
        .iter()
        .any(|segment| matches!(segment, ProxySegment::ToolUse { .. }));
    for segment in segments {
        match segment {
            ProxySegment::Text(text) => {
                if !text.is_empty() && should_render_proxy_text_segment(&text, has_tool_use) {
                    write!(out, "{text}")
                        .and_then(|_| out.flush())
                        .map_err(|error| RuntimeError::new(error.to_string()))?;
                    events.push(AssistantEvent::TextDelta(text));
                }
            }
            ProxySegment::ToolUse { id, name, input } => {
                events.push(AssistantEvent::ToolUse { id, name, input });
            }
        }
    }
    Ok(())
}

fn should_render_proxy_text_segment(text: &str, has_tool_use: bool) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }

    if contains_proxy_markup(trimmed) {
        return false;
    }

    if has_tool_use {
        let normalized = trimmed.to_ascii_lowercase();
        let boilerplate = [
            "now let me",
            "let me",
            "i'll",
            "i will",
            "first, i'll",
            "first i will",
            "first i'll",
            "creating",
            "writing",
            "saving",
            "updating",
            "reading",
            "editing",
        ];
        if boilerplate
            .iter()
            .any(|prefix| normalized.starts_with(prefix))
        {
            return false;
        }
    }

    true
}

fn contains_proxy_markup(text: &str) -> bool {
    let normalized = text.to_ascii_lowercase();
    let markers = [
        "<tool_call",
        "</tool_call",
        "<tool_result",
        "</tool_result",
        "<arg",
        "</arg",
        "</parameter",
        "<read_file",
        "<write_file",
        "<edit_file",
        "<bash",
        "<glob_search",
        "<grep_search",
        "<webfetch",
        "<websearch",
        "<todowrite",
        "<skill",
        "<agent",
        "<toolsearch",
        "<notebookedit",
        "<sleep",
        "<powershell",
    ];
    markers.iter().any(|marker| normalized.contains(marker))
}

struct CliToolExecutor {
    service: ApiService,
    tool_registry: GlobalToolRegistry,
    mcp_catalog: McpCatalog,
    tool_specs: Vec<RuntimeToolSpec>,
    allowed_tools: Option<AllowedToolSet>,
    emit_output: bool,
}

impl CliToolExecutor {
    fn new(
        service: ApiService,
        tool_registry: GlobalToolRegistry,
        mcp_catalog: McpCatalog,
        tool_specs: Vec<RuntimeToolSpec>,
        allowed_tools: Option<AllowedToolSet>,
        emit_output: bool,
    ) -> Self {
        Self {
            service,
            tool_registry,
            mcp_catalog,
            tool_specs,
            allowed_tools,
            emit_output,
        }
    }
}

impl ToolExecutor for CliToolExecutor {
    fn execute(&mut self, tool_name: &str, input: &str) -> Result<String, ToolError> {
        if self
            .allowed_tools
            .as_ref()
            .is_some_and(|allowed| !allowed.contains(tool_name))
        {
            return Err(ToolError::new(format!(
                "tool `{tool_name}` is not enabled by the current --allowedTools setting"
            )));
        }
        let value = parse_tool_input_value(tool_name, input, &self.tool_specs)?;
        let _service_guard = set_active_backend_service(self.service);
        let output = if let Some(tool) = self.mcp_catalog.find_tool(tool_name) {
            call_mcp_tool(tool, &value).map_err(|error| ToolError::new(error.to_string()))
        } else {
            self.tool_registry
                .execute(tool_name, &value)
                .map_err(ToolError::new)
        }?;
        if self.emit_output {
            let block = render_tool_result_block(tool_name, &output);
            if !block.is_empty() {
                let mut stdout = io::stdout();
                write!(stdout, "{block}")
                    .and_then(|_| stdout.flush())
                    .map_err(|error| ToolError::new(error.to_string()))?;
            }
        }
        Ok(output)
    }
}

/// Render the compact, already-ANSI-styled result block that visually hangs
/// off the tool-call header we printed when the model invoked the tool.
///
/// Where possible we use [`render_structured_tool_preview`] to produce a
/// hand-crafted summary for the tool family. When the output is unknown we
/// fall back to a single "N bytes / N lines" one-liner so the transcript
/// stays compact. Full payloads are always retained in conversation context
/// regardless of what the TUI shows.
fn render_tool_result_block(tool_name: &str, output: &str) -> String {
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

fn parse_tool_input_value(
    tool_name: &str,
    input: &str,
    tool_specs: &[RuntimeToolSpec],
) -> Result<serde_json::Value, ToolError> {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(input) {
        return Ok(value);
    }

    if let Some(value) = extract_first_json_object(input) {
        return Ok(value);
    }

    if input.contains("<tool_call") {
        let tool_use = parse_proxy_response(input, tool_specs)
            .map_err(ToolError::new)?
            .into_iter()
            .find_map(|segment| match segment {
                ProxySegment::ToolUse { name, input, .. } if name == tool_name => Some(input),
                _ => None,
            })
            .ok_or_else(|| ToolError::new("proxy tool call did not contain a matching tool"))?;
        return serde_json::from_str(&tool_use).map_err(|error| {
            ToolError::new(format!("invalid recovered proxy tool JSON: {error}"))
        });
    }

    if input.contains("<arg") {
        let wrapped = format!("<tool_call name=\"{tool_name}\">{input}</tool_call>");
        let tool_use = parse_proxy_response(&wrapped, tool_specs)
            .map_err(ToolError::new)?
            .into_iter()
            .find_map(|segment| match segment {
                ProxySegment::ToolUse { input, .. } => Some(input),
                _ => None,
            })
            .ok_or_else(|| ToolError::new("proxy arg fragment did not produce tool input"))?;
        return serde_json::from_str(&tool_use)
            .map_err(|error| ToolError::new(format!("invalid recovered proxy arg JSON: {error}")));
    }

    Err(ToolError::new(format!(
        "invalid tool input JSON: could not parse {input:?}"
    )))
}

fn extract_first_json_object(input: &str) -> Option<serde_json::Value> {
    let start = input.find('{')?;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (offset, ch) in input[start..].char_indices() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    let end = start + offset + ch.len_utf8();
                    return serde_json::from_str(&input[start..end]).ok();
                }
            }
            _ => {}
        }
    }

    None
}

fn permission_policy(mode: PermissionMode, tool_specs: &[RuntimeToolSpec]) -> PermissionPolicy {
    tool_specs
        .iter()
        .fold(PermissionPolicy::new(mode), |policy, spec| {
            policy.with_tool_requirement(spec.name.clone(), spec.required_permission)
        })
}

fn convert_messages(
    messages: &[ConversationMessage],
    service: ApiService,
) -> Result<Vec<InputMessage>, RuntimeError> {
    let cwd = env::current_dir().map_err(|error| {
        RuntimeError::new(format!("failed to resolve current directory: {error}"))
    })?;
    sanitize_messages_for_api(messages)
        .iter()
        .filter_map(|message| {
            let role = match message.role {
                MessageRole::System | MessageRole::User | MessageRole::Tool => "user",
                MessageRole::Assistant => "assistant",
            };
            let content = message
                .blocks
                .iter()
                .try_fold(Vec::new(), |mut acc, block| {
                    match block {
                        ContentBlock::Text { text } => {
                            if message.role == MessageRole::User {
                                acc.extend(
                                    prompt_to_content_blocks(text, &cwd)
                                        .map_err(RuntimeError::new)?,
                                );
                            } else {
                                acc.push(InputContentBlock::Text { text: text.clone() });
                            }
                        }
                        ContentBlock::Thinking { .. } => {}
                        ContentBlock::ToolUse { id, name, input } => {
                            acc.push(InputContentBlock::ToolUse {
                                id: id.clone(),
                                name: name.clone(),
                                input: serde_json::from_str(input)
                                    .unwrap_or_else(|_| serde_json::json!({ "raw": input })),
                            })
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            output,
                            is_error,
                            ..
                        } => acc.push(InputContentBlock::ToolResult {
                            tool_use_id: tool_use_id.clone(),
                            content: vec![ToolResultContentBlock::Text {
                                text: output.clone(),
                            }],
                            is_error: *is_error,
                        }),
                    }
                    Ok::<_, RuntimeError>(acc)
                });
            let reasoning_content =
                if service == ApiService::OpencodeGo && message.role == MessageRole::Assistant {
                    let reasoning = message
                        .blocks
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::Thinking { text, .. } if !text.is_empty() => {
                                Some(text.as_str())
                            }
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n\n");
                    (!reasoning.is_empty()).then_some(reasoning)
                } else {
                    None
                };
            match content {
                Ok(content) if !content.is_empty() => Some(Ok(InputMessage {
                    role: role.to_string(),
                    content,
                    reasoning_content: reasoning_content.clone(),
                    reasoning: reasoning_content,
                })),
                Ok(_) => None,
                Err(error) => Some(Err(error)),
            }
        })
        .collect()
}

fn sanitize_messages_for_api(messages: &[ConversationMessage]) -> Vec<ConversationMessage> {
    let tool_use_ids = messages
        .iter()
        .flat_map(|message| message.blocks.iter())
        .filter_map(|block| match block {
            ContentBlock::ToolUse { id, .. } => Some(id.clone()),
            _ => None,
        })
        .collect::<HashSet<_>>();
    let tool_result_ids = messages
        .iter()
        .flat_map(|message| message.blocks.iter())
        .filter_map(|block| match block {
            ContentBlock::ToolResult { tool_use_id, .. } => Some(tool_use_id.clone()),
            _ => None,
        })
        .collect::<HashSet<_>>();
    let balanced_tool_ids = tool_use_ids
        .intersection(&tool_result_ids)
        .cloned()
        .collect::<HashSet<_>>();

    messages
        .iter()
        .filter_map(|message| {
            let blocks = message
                .blocks
                .iter()
                .filter(|block| match block {
                    ContentBlock::ToolUse { id, .. } => balanced_tool_ids.contains(id),
                    ContentBlock::ToolResult { tool_use_id, .. } => {
                        balanced_tool_ids.contains(tool_use_id)
                    }
                    _ => true,
                })
                .cloned()
                .collect::<Vec<_>>();
            if blocks.is_empty() {
                return None;
            }
            Some(ConversationMessage {
                role: message.role,
                blocks,
                usage: message.usage,
            })
        })
        .collect()
}

fn prompt_to_content_blocks(input: &str, cwd: &Path) -> Result<Vec<InputContentBlock>, String> {
    let mut blocks = Vec::new();
    let mut text_buffer = String::new();
    let mut chars = input.char_indices().peekable();

    while let Some((index, ch)) = chars.next() {
        if ch == '!' && input[index..].starts_with("![") {
            if let Some((_, path_start, path_end)) = parse_markdown_image_ref(input, index) {
                flush_text_block(&mut blocks, &mut text_buffer);
                let path = &input[path_start..path_end];
                blocks.push(load_image_block(path, cwd)?);
                while let Some((next_index, _)) = chars.peek() {
                    if *next_index < path_end + 1 {
                        let _ = chars.next();
                    } else {
                        break;
                    }
                }
                continue;
            }
        }

        if ch == '@' && is_ref_boundary(input[..index].chars().next_back()) {
            let path_end = find_path_end(input, index + 1);
            if path_end > index + 1 {
                let candidate = &input[index + 1..path_end];
                if looks_like_image_ref(candidate, cwd) {
                    flush_text_block(&mut blocks, &mut text_buffer);
                    blocks.push(load_image_block(candidate, cwd)?);
                    while let Some((next_index, _)) = chars.peek() {
                        if *next_index < path_end {
                            let _ = chars.next();
                        } else {
                            break;
                        }
                    }
                    continue;
                }
            }
        }

        text_buffer.push(ch);
    }

    flush_text_block(&mut blocks, &mut text_buffer);
    if blocks.is_empty() {
        blocks.push(InputContentBlock::Text {
            text: input.to_string(),
        });
    }
    Ok(blocks)
}

fn parse_markdown_image_ref(input: &str, start: usize) -> Option<(usize, usize, usize)> {
    let after_bang = input.get(start + 2..)?;
    let alt_end_offset = after_bang.find("](")?;
    let path_start = start + 2 + alt_end_offset + 2;
    let remainder = input.get(path_start..)?;
    let path_end_offset = remainder.find(')')?;
    let path_end = path_start + path_end_offset;
    Some((start + 2 + alt_end_offset, path_start, path_end))
}

fn is_ref_boundary(ch: Option<char>) -> bool {
    ch.is_none_or(char::is_whitespace)
}

fn find_path_end(input: &str, start: usize) -> usize {
    input[start..]
        .char_indices()
        .find_map(|(offset, ch)| ch.is_whitespace().then_some(start + offset))
        .unwrap_or(input.len())
}

fn looks_like_image_ref(candidate: &str, cwd: &Path) -> bool {
    let resolved = resolve_prompt_path(candidate, cwd);
    media_type_for_path(Path::new(candidate)).is_some()
        || resolved.is_file()
        || candidate.contains(std::path::MAIN_SEPARATOR)
        || candidate.starts_with("./")
        || candidate.starts_with("../")
}

fn flush_text_block(blocks: &mut Vec<InputContentBlock>, text_buffer: &mut String) {
    if text_buffer.is_empty() {
        return;
    }
    blocks.push(InputContentBlock::Text {
        text: std::mem::take(text_buffer),
    });
}

fn load_image_block(path_ref: &str, cwd: &Path) -> Result<InputContentBlock, String> {
    let resolved = resolve_prompt_path(path_ref, cwd);
    let media_type = media_type_for_path(&resolved).ok_or_else(|| {
        format!(
            "unsupported image format for reference {IMAGE_REF_PREFIX}{path_ref}; supported: png, jpg, jpeg, gif, webp"
        )
    })?;
    let bytes = fs::read(&resolved).map_err(|error| {
        format!(
            "failed to read image reference {}: {error}",
            resolved.display()
        )
    })?;
    Ok(InputContentBlock::Image {
        source: ImageSource {
            kind: "base64".to_string(),
            media_type: media_type.to_string(),
            data: encode_base64(&bytes),
        },
    })
}

fn resolve_prompt_path(path_ref: &str, cwd: &Path) -> PathBuf {
    let path = Path::new(path_ref);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    }
}

fn media_type_for_path(path: &Path) -> Option<&'static str> {
    let extension = path.extension()?.to_str()?.to_ascii_lowercase();
    match extension.as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        _ => None,
    }
}

fn encode_base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut output = String::new();
    let mut index = 0;
    while index + 3 <= bytes.len() {
        let block = (u32::from(bytes[index]) << 16)
            | (u32::from(bytes[index + 1]) << 8)
            | u32::from(bytes[index + 2]);
        output.push(TABLE[((block >> 18) & 0x3F) as usize] as char);
        output.push(TABLE[((block >> 12) & 0x3F) as usize] as char);
        output.push(TABLE[((block >> 6) & 0x3F) as usize] as char);
        output.push(TABLE[(block & 0x3F) as usize] as char);
        index += 3;
    }

    match bytes.len().saturating_sub(index) {
        1 => {
            let block = u32::from(bytes[index]) << 16;
            output.push(TABLE[((block >> 18) & 0x3F) as usize] as char);
            output.push(TABLE[((block >> 12) & 0x3F) as usize] as char);
            output.push('=');
            output.push('=');
        }
        2 => {
            let block = (u32::from(bytes[index]) << 16) | (u32::from(bytes[index + 1]) << 8);
            output.push(TABLE[((block >> 18) & 0x3F) as usize] as char);
            output.push(TABLE[((block >> 12) & 0x3F) as usize] as char);
            output.push(TABLE[((block >> 6) & 0x3F) as usize] as char);
            output.push('=');
        }
        _ => {}
    }

    output
}

fn convert_proxy_messages_to_input_messages(messages: Vec<ProxyMessage>) -> Vec<InputMessage> {
    messages
        .into_iter()
        .map(|message| InputMessage {
            role: message.role,
            content: vec![InputContentBlock::Text {
                text: message.content,
            }],
            reasoning_content: None,
            reasoning: None,
        })
        .collect()
}

fn effective_reasoning_effort(
    collaboration_mode: CollaborationMode,
    reasoning_effort: Option<ReasoningEffort>,
) -> Option<ReasoningEffort> {
    reasoning_effort.or(match collaboration_mode {
        CollaborationMode::Build => None,
        CollaborationMode::Plan => Some(ReasoningEffort::Medium),
    })
}

fn reasoning_effort_label(reasoning_effort: Option<ReasoningEffort>) -> &'static str {
    match reasoning_effort {
        Some(ReasoningEffort::Minimal) => "minimal",
        Some(ReasoningEffort::Low) => "low",
        Some(ReasoningEffort::Medium) => "medium",
        Some(ReasoningEffort::High) => "high",
        Some(ReasoningEffort::XHigh) => "xhigh",
        None => "default",
    }
}

fn format_mode_report(collaboration_mode: CollaborationMode) -> String {
    format!(
        "Mode\n  Active mode      {}\n  Toggle           press Tab on an empty prompt or use /mode build|plan",
        collaboration_mode.as_str()
    )
}

fn format_mode_switch_report(collaboration_mode: CollaborationMode) -> String {
    format!(
        "Mode updated\n  Result           {}\n  Applies to       subsequent requests",
        collaboration_mode.as_str()
    )
}

fn format_reasoning_report(
    configured: Option<ReasoningEffort>,
    effective: Option<ReasoningEffort>,
) -> String {
    format!(
        "Reasoning\n  Configured       {}\n  Effective        {}\n\nUsage\n  Inspect with     /reasoning\n  Set with         /reasoning default|minimal|low|medium|high|xhigh",
        reasoning_effort_label(configured),
        reasoning_effort_label(effective),
    )
}

fn format_reasoning_switch_report(
    configured: Option<ReasoningEffort>,
    effective: Option<ReasoningEffort>,
) -> String {
    format!(
        "Reasoning updated\n  Configured       {}\n  Effective        {}\n  Applies to       subsequent requests",
        reasoning_effort_label(configured),
        reasoning_effort_label(effective),
    )
}

fn format_fast_mode_report(fast_mode: FastMode) -> String {
    format!(
        "Fast Mode\n  Active mode      {}\n\nUsage\n  Inspect with     /fast\n  Toggle with      /fast on or /fast off",
        fast_mode.as_str()
    )
}

fn format_fast_mode_switch_report(fast_mode: FastMode) -> String {
    format!(
        "Fast mode updated\n  Result           {}\n  Applies to       subsequent requests",
        fast_mode.as_str()
    )
}

/// Derive a compact model label for the prompt status strip. Model IDs can
/// be very long (e.g. `anthropic/claude-opus-4-20250514`); we trim the
/// provider-style prefix so the strip stays readable on narrow terminals.
fn short_model_name(model: &str) -> &str {
    model.rsplit('/').next().unwrap_or(model)
}

fn collect_tool_uses(summary: &runtime::TurnSummary) -> Vec<JsonValue> {
    summary
        .assistant_messages
        .iter()
        .flat_map(|message| message.blocks.iter())
        .filter_map(|block| match block {
            ContentBlock::ToolUse { id, name, input } => Some(serde_json::json!({
                "id": id,
                "name": name,
                "input": input,
            })),
            _ => None,
        })
        .collect()
}

fn collect_tool_results(summary: &runtime::TurnSummary) -> Vec<JsonValue> {
    summary
        .tool_results
        .iter()
        .flat_map(|message| message.blocks.iter())
        .filter_map(|block| match block {
            ContentBlock::ToolResult {
                tool_use_id,
                tool_name,
                output,
                is_error,
            } => Some(serde_json::json!({
                "tool_use_id": tool_use_id,
                "tool_name": tool_name,
                "output": output,
                "is_error": is_error,
            })),
            _ => None,
        })
        .collect()
}

fn run_resume_command(
    session_path: &Path,
    session: &Session,
    command: &SlashCommand,
) -> Result<ResumeCommandOutcome, Box<dyn std::error::Error>> {
    match command {
        SlashCommand::Help { topic } => Ok(ResumeCommandOutcome {
            session: session.clone(),
            message: Some(topic.as_deref().map_or_else(render_repl_help, |topic| {
                render_slash_command_help_topic(Some(topic))
            })),
        }),
        SlashCommand::Compact => {
            let result = runtime::compact_session(
                session,
                CompactionConfig {
                    max_estimated_tokens: 0,
                    ..CompactionConfig::default()
                },
            );
            let removed = result.removed_message_count;
            let kept = result.compacted_session.messages.len();
            let skipped = removed == 0;
            Ok(ResumeCommandOutcome {
                session: result.compacted_session,
                message: Some(if skipped {
                    format!(
                        "Compact\n  Result           skipped\n  Reason           session below compaction threshold\n  Messages kept    {kept}"
                    )
                } else {
                    format!(
                        "Compact\n  Result           compacted\n  Messages removed {removed}\n  Messages kept    {kept}"
                    )
                }),
            })
        }
        SlashCommand::Clear { confirm } => {
            if !confirm {
                return Ok(ResumeCommandOutcome {
                    session: session.clone(),
                    message: Some(
                        "clear: confirmation required; rerun with /clear --confirm".to_string(),
                    ),
                });
            }
            let cleared = Session::new();
            Ok(ResumeCommandOutcome {
                session: cleared,
                message: Some(format!(
                    "Cleared resumed session file {}.",
                    session_path.display()
                )),
            })
        }
        SlashCommand::Status => {
            let tracker = UsageTracker::from_session(session);
            let usage = tracker.cumulative_usage();
            let state = session_runtime_state(
                session,
                DEFAULT_MODEL,
                None,
                default_permission_mode(),
                CollaborationMode::Build,
                None,
                FastMode::Off,
                false,
            );
            Ok(ResumeCommandOutcome {
                session: session.clone(),
                message: Some(format_status_report(
                    state.service,
                    &state.model,
                    StatusUsage {
                        message_count: session.messages.len(),
                        turns: tracker.turns(),
                        latest: tracker.current_turn_usage(),
                        cumulative: usage,
                        estimated_tokens: 0,
                    },
                    state.permission_mode.as_str(),
                    provider_label_for_service_model(state.service, &state.model).as_deref(),
                    state.proxy_tool_calls,
                    state.collaboration_mode,
                    effective_reasoning_effort(state.collaboration_mode, state.reasoning_effort),
                    state.fast_mode,
                    &McpCatalog::default(),
                    &status_context(Some(session_path))?,
                )),
            })
        }
        SlashCommand::Config { section } => Ok(ResumeCommandOutcome {
            session: session.clone(),
            message: Some(render_config_report(section.as_deref())?),
        }),
        SlashCommand::Memory => Ok(ResumeCommandOutcome {
            session: session.clone(),
            message: Some(render_memory_report()?),
        }),
        SlashCommand::Init => {
            let state = session_runtime_state(
                session,
                DEFAULT_MODEL,
                None,
                default_permission_mode(),
                CollaborationMode::Build,
                None,
                FastMode::Off,
                false,
            );
            let (report, warning) =
                initialize_repo_for_model(&env::current_dir()?, state.service, &state.model)?;
            let mut message = String::new();
            if let Some(warning) = warning {
                writeln!(&mut message, "{warning}")?;
                writeln!(&mut message)?;
            }
            message.push_str(&report.render());
            Ok(ResumeCommandOutcome {
                session: session.clone(),
                message: Some(message),
            })
        }
        SlashCommand::Diff => Ok(ResumeCommandOutcome {
            session: session.clone(),
            message: Some(render_diff_report()?),
        }),
        SlashCommand::Version => Ok(ResumeCommandOutcome {
            session: session.clone(),
            message: Some(render_version_report()),
        }),
        SlashCommand::Export { path } => {
            let export_path = resolve_export_path(path.as_deref(), session)?;
            fs::write(&export_path, render_export_text(session))?;
            Ok(ResumeCommandOutcome {
                session: session.clone(),
                message: Some(format!(
                    "Export\n  Result           wrote transcript\n  File             {}\n  Messages         {}",
                    export_path.display(),
                    session.messages.len(),
                )),
            })
        }
        SlashCommand::Resume { .. }
        | SlashCommand::Model { .. }
        | SlashCommand::Logout { .. }
        | SlashCommand::Mcp { .. }
        | SlashCommand::Permissions { .. }
        | SlashCommand::Reasoning { .. }
        | SlashCommand::Fast { .. }
        | SlashCommand::Mode { .. }
        | SlashCommand::Session { .. }
        | SlashCommand::Sessions
        | SlashCommand::Branch { .. }
        | SlashCommand::Worktree { .. }
        | SlashCommand::Plugins { .. }
        | SlashCommand::Agents { .. }
        | SlashCommand::Skills { .. }
        | SlashCommand::Unknown(_) => Err("unsupported resumed slash command".into()),
    }
}

fn print_help() {
    println!("Pebble");
    println!();
    println!("Usage");
    println!(
        "  pebble [--model MODEL] [--permission-mode MODE] [--mode MODE] [--reasoning LEVEL] [--fast] [--allowedTools TOOL[,TOOL...]]"
    );
    println!("                                               Start interactive REPL");
    println!(
        "  pebble login [SERVICE] [--api-key KEY]    Save credentials for NanoGPT, Synthetic, OpenAI Codex, OpenCode Go, or Exa"
    );
    println!(
        "                                               Services: nanogpt, synthetic, openai-codex, opencode-go, exa"
    );
    println!("  pebble logout [SERVICE]                   Remove saved credentials for a service");
    println!("  pebble model [MODEL_ID]                   Choose or persist a default model");
    println!("  pebble provider [PROVIDER_ID|default]     Choose a provider for the active model");
    println!("  pebble proxy [on|off|status]              Toggle XML tool-call proxy mode");
    println!(
        "  pebble mcp [status|tools|reload|add <name>|enable <name>|disable <name>] Inspect configured MCP servers and tools"
    );
    println!("  pebble plugins [list|help|install|enable|disable|uninstall|update] [TARGET]");
    println!("  pebble branch [list|create|switch] [ARG]  Inspect or change git branches");
    println!("  pebble worktree [list|add|remove|prune]   Inspect or manage git worktrees");
    println!("  pebble agents [list|help]                 List configured Pebble agents");
    println!("  pebble skills [list|help|init <name>]     List or scaffold Pebble skills");
    println!("  pebble init                               Create starter Pebble project files");
    println!("  pebble doctor                             Run local environment diagnostics");
    println!("  pebble self-update                        Update from GitHub releases");
    println!("  pebble resume [SESSION_ID_OR_PATH]");
    println!("                                               Resume a saved session, or pick one and enter the REPL");
    println!("  pebble --resume [SESSION_ID_OR_PATH] [/status] [/compact] [...]");
    println!("                                               Resume a saved session and optionally run slash commands");
    println!(
        "  pebble prompt [--model MODEL] [--permission-mode MODE] [--mode MODE] [--reasoning LEVEL] [--fast] [--output-format text|json] TEXT"
    );
    println!(
        "                                               Send one prompt and stream the response"
    );
    println!(
        "  pebble [--model MODEL] [--permission-mode MODE] [--mode MODE] [--reasoning LEVEL] [--fast] [--output-format text|json] TEXT"
    );
    println!(
        "                                               Shorthand non-interactive prompt mode"
    );
    println!("  pebble dump-manifests");
    println!("  pebble bootstrap-plan");
    println!("  pebble system-prompt [--cwd PATH] [--date YYYY-MM-DD]");
    println!("  pebble --version");
    println!(
        "  --permission-mode MODE                     read-only, workspace-write, or danger-full-access"
    );
    println!("  --mode MODE                               build or plan");
    println!(
        "  --reasoning LEVEL                         default, minimal, low, medium, high, or xhigh"
    );
    println!(
        "  --fast                                     Enable ChatGPT fast mode for OpenAI Codex"
    );
    println!(
        "  --thinking                                 Compatibility alias for --reasoning medium"
    );
    println!(
        "  --output-format FORMAT                     Non-interactive output format: text or json"
    );
    println!();
    println!("{}", render_repl_help());
    println!();
    println!("{}", render_help_topics_overview());
}

fn print_version() {
    println!("{}", render_version_report());
}

fn run_init() -> Result<(), Box<dyn std::error::Error>> {
    let model = default_model_or(DEFAULT_MODEL);
    run_init_with_model(infer_service_for_model(&model), &model)
}

fn run_init_with_model(service: ApiService, model: &str) -> Result<(), Box<dyn std::error::Error>> {
    let cwd = env::current_dir()?;
    let (report, warning) = initialize_repo_for_model(&cwd, service, model)?;
    if let Some(warning) = warning {
        eprintln!("{warning}");
    }
    println!("{}", report.render());
    Ok(())
}

fn initialize_repo_for_model(
    cwd: &Path,
    service: ApiService,
    model: &str,
) -> Result<(crate::init::InitReport, Option<String>), Box<dyn std::error::Error>> {
    if cwd.join("PEBBLE.md").exists() {
        return Ok((initialize_repo(cwd)?, None));
    }

    match generate_pebble_md(cwd, service, model) {
        Ok(content) => Ok((initialize_repo_with_pebble_md(cwd, &content)?, None)),
        Err(error) => Ok((
            initialize_repo(cwd)?,
            Some(format_init_generation_warning(service, model, &*error)),
        )),
    }
}

fn generate_pebble_md(
    cwd: &Path,
    service: ApiService,
    model: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let request = MessageRequest {
        model: model.to_string(),
        max_tokens: INIT_PEBBLE_MD_MAX_TOKENS,
        messages: vec![InputMessage::user_text(build_init_generation_prompt(cwd)?)],
        system: Some(init_generation_system_prompt().to_string()),
        tools: None,
        tool_choice: None,
        thinking: None,
        reasoning_effort: None,
        fast_mode: false,
        stream: false,
    };
    let mut client = NanoGptClient::from_service_env(service)?;
    if service == ApiService::NanoGpt {
        client = client.with_provider(provider_for_model(model));
    }

    let runtime = tokio::runtime::Runtime::new()?;
    let response = runtime.block_on(async { client.send_message(&request).await })?;
    extract_generated_pebble_md(response)
}

fn init_generation_system_prompt() -> &'static str {
    "You write repo-specific PEBBLE.md files for a coding assistant. Return only markdown for the file with no code fences or prefatory text. Stay concrete, concise, and factual. Do not invent commands, directories, workflows, or architecture details. If a detail is uncertain from the provided context, say to verify it or omit it."
}

fn build_init_generation_prompt(cwd: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let project_context = runtime::ProjectContext::discover_with_git(cwd, DEFAULT_DATE)?;
    let mut prompt = String::new();
    writeln!(
        &mut prompt,
        "Create a project-specific `PEBBLE.md` for this repository."
    )?;
    writeln!(
        &mut prompt,
        "Use the supplied repository context to replace the generic starter template with concrete guidance."
    )?;
    writeln!(&mut prompt)?;
    writeln!(&mut prompt, "Required output shape:")?;
    writeln!(&mut prompt, "- `# PEBBLE.md`")?;
    writeln!(&mut prompt, "- `## Project Overview`")?;
    writeln!(&mut prompt, "- `## Repository Shape`")?;
    writeln!(&mut prompt, "- `## Commands`")?;
    writeln!(&mut prompt, "- `## Working Agreement`")?;
    writeln!(&mut prompt)?;
    writeln!(&mut prompt, "Instructions:")?;
    writeln!(
        &mut prompt,
        "- Keep it concise, actionable, and specific to this repo."
    )?;
    writeln!(
        &mut prompt,
        "- Prefer bullet lists for commands and operational guidance."
    )?;
    writeln!(
        &mut prompt,
        "- Mention verification commands only when supported by the provided files."
    )?;
    writeln!(
        &mut prompt,
        "- If the context is incomplete, say what to verify instead of guessing."
    )?;
    writeln!(&mut prompt)?;
    writeln!(&mut prompt, "Repository context")?;
    writeln!(&mut prompt, "Working directory: {}", cwd.display())?;
    writeln!(
        &mut prompt,
        "Top-level entries:\n{}",
        render_init_top_level_entries(cwd)?
    )?;

    if let Some(git_status) = project_context.git_status.as_deref() {
        let trimmed = git_status.trim();
        if !trimmed.is_empty() {
            writeln!(&mut prompt)?;
            writeln!(&mut prompt, "Git status:")?;
            writeln!(&mut prompt, "```text")?;
            writeln!(&mut prompt, "{trimmed}")?;
            writeln!(&mut prompt, "```")?;
        }
    }

    let context_files = collect_init_context_files(cwd, &project_context);
    if !context_files.is_empty() {
        writeln!(&mut prompt)?;
        writeln!(&mut prompt, "Key file excerpts:")?;
        for path in context_files {
            if let Some(snippet) = read_init_context_file(&path) {
                writeln!(&mut prompt, "### {}", display_init_context_path(cwd, &path))?;
                writeln!(&mut prompt, "```text")?;
                writeln!(&mut prompt, "{snippet}")?;
                writeln!(&mut prompt, "```")?;
            }
        }
    }

    writeln!(&mut prompt)?;
    writeln!(&mut prompt, "Starter template to improve:")?;
    writeln!(&mut prompt, "```markdown")?;
    writeln!(&mut prompt, "{}", render_init_pebble_md(cwd))?;
    writeln!(&mut prompt, "```")?;

    Ok(prompt)
}

fn render_init_top_level_entries(cwd: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let mut entries = fs::read_dir(cwd)?
        .flatten()
        .map(|entry| {
            let path = entry.path();
            let mut name = entry.file_name().to_string_lossy().into_owned();
            if path.is_dir() {
                name.push('/');
            }
            name
        })
        .collect::<Vec<_>>();
    entries.sort();

    let remaining = entries.len().saturating_sub(MAX_INIT_TOP_LEVEL_ENTRIES);
    entries.truncate(MAX_INIT_TOP_LEVEL_ENTRIES);
    if remaining > 0 {
        entries.push(format!("... and {remaining} more"));
    }

    Ok(entries
        .into_iter()
        .map(|entry| format!("- {entry}"))
        .collect::<Vec<_>>()
        .join("\n"))
}

fn collect_init_context_files(
    cwd: &Path,
    project_context: &runtime::ProjectContext,
) -> Vec<PathBuf> {
    let mut seen = HashSet::new();
    let mut paths = Vec::new();
    for path in [
        cwd.join("README.md"),
        cwd.join("Cargo.toml"),
        cwd.join("package.json"),
        cwd.join("pyproject.toml"),
        cwd.join("go.mod"),
        cwd.join("Makefile"),
        cwd.join("justfile"),
        cwd.join("Justfile"),
    ] {
        if path.is_file() && seen.insert(path.clone()) {
            paths.push(path);
        }
    }
    for file in &project_context.instruction_files {
        if let Some(name) = file.path.file_name().and_then(|name| name.to_str()) {
            if name.eq_ignore_ascii_case("PEBBLE.md") {
                continue;
            }
        }
        if file.path.is_file() && seen.insert(file.path.clone()) {
            paths.push(file.path.clone());
        }
    }
    paths.truncate(MAX_INIT_CONTEXT_FILES);
    paths
}

fn read_init_context_file(path: &Path) -> Option<String> {
    let contents = fs::read_to_string(path).ok()?;
    let normalized = contents.replace("\r\n", "\n");
    let trimmed = normalized.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(truncate_chars(trimmed, MAX_INIT_CONTEXT_CHARS))
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    let mut truncated = text.chars().take(max_chars).collect::<String>();
    if text.chars().count() > max_chars {
        truncated.push_str("\n... [truncated]");
    }
    truncated
}

fn display_init_context_path(cwd: &Path, path: &Path) -> String {
    path.strip_prefix(cwd).unwrap_or(path).display().to_string()
}

fn extract_generated_pebble_md(
    response: MessageResponse,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut text = String::new();
    for block in response.content {
        if let OutputContentBlock::Text { text: block_text } = block {
            text.push_str(&block_text);
        }
    }

    normalize_generated_pebble_md(&text)
        .ok_or_else(|| "model returned no markdown content for PEBBLE.md".into())
}

fn normalize_generated_pebble_md(raw: &str) -> Option<String> {
    let normalized = raw.replace("\r\n", "\n");
    let trimmed = normalized.trim();
    if trimmed.is_empty() {
        return None;
    }

    let unfenced = strip_markdown_code_fence(trimmed).unwrap_or(trimmed).trim();
    if unfenced.is_empty() {
        return None;
    }

    let mut content = unfenced.find("# PEBBLE.md").map_or_else(
        || unfenced.to_string(),
        |index| unfenced[index..].to_string(),
    );
    if !content.starts_with("# PEBBLE.md") {
        content = format!("# PEBBLE.md\n\n{content}");
    }
    if !content.ends_with('\n') {
        content.push('\n');
    }
    Some(content)
}

fn strip_markdown_code_fence(text: &str) -> Option<&str> {
    let rest = text.strip_prefix("```")?;
    let newline = rest.find('\n')?;
    let body = &rest[newline + 1..];
    body.strip_suffix("\n```")
        .or_else(|| body.strip_suffix("```"))
}

fn format_init_generation_warning(
    service: ApiService,
    model: &str,
    error: &dyn std::fmt::Display,
) -> String {
    format!(
        "warning: failed to generate a repo-specific PEBBLE.md with {}/{}; used the starter template instead: {error}",
        service.display_name(),
        model
    )
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
struct GitHubRelease {
    tag_name: String,
    #[serde(default)]
    body: String,
    #[serde(default)]
    assets: Vec<GitHubReleaseAsset>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
struct GitHubReleaseAsset {
    name: String,
    browser_download_url: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SelectedReleaseAssets {
    binary: GitHubReleaseAsset,
    checksum: GitHubReleaseAsset,
}

fn run_self_update() -> Result<(), Box<dyn std::error::Error>> {
    let Some(release) = fetch_latest_release()? else {
        println!(
            "{}",
            render_update_report(
                "No published release available",
                Some(VERSION),
                None,
                Some("GitHub latest release endpoint returned no published release for nanogpt-community/pebble."),
                None,
            )
        );
        return Ok(());
    };

    let latest_version = normalize_version_tag(&release.tag_name);
    if !is_newer_version(VERSION, &latest_version) {
        println!(
            "{}",
            render_update_report(
                "Already up to date",
                Some(VERSION),
                Some(&latest_version),
                Some("Current binary already matches the latest published release."),
                Some(&release.body),
            )
        );
        return Ok(());
    }

    let selected = match select_release_assets(&release) {
        Ok(selected) => selected,
        Err(message) => {
            println!(
                "{}",
                render_update_report(
                    "Release found, but no installable asset matched this platform",
                    Some(VERSION),
                    Some(&latest_version),
                    Some(&message),
                    Some(&release.body),
                )
            );
            return Ok(());
        }
    };

    #[cfg(windows)]
    {
        println!(
            "{}",
            render_update_report(
                "Manual install required on Windows",
                Some(VERSION),
                Some(&latest_version),
                Some(&format!(
                    "Download {} from the latest GitHub release and replace pebble.exe manually. In-place self-update is not supported on Windows yet.",
                    selected.binary.name
                )),
                Some(&release.body),
            )
        );
        return Ok(());
    }

    #[cfg(not(windows))]
    {
        let client = build_self_update_client()?;
        let binary_bytes = download_bytes(&client, &selected.binary.browser_download_url)?;
        let checksum_manifest = download_text(&client, &selected.checksum.browser_download_url)?;
        let expected_checksum = parse_checksum_for_asset(&checksum_manifest, &selected.binary.name)
            .ok_or_else(|| {
                format!(
                    "checksum manifest did not contain an entry for {}",
                    selected.binary.name
                )
            })?;
        let actual_checksum = sha256_hex(&binary_bytes);
        if actual_checksum != expected_checksum {
            return Err(format!(
                "downloaded asset checksum mismatch for {} (expected {}, got {})",
                selected.binary.name, expected_checksum, actual_checksum
            )
            .into());
        }

        replace_current_executable(&binary_bytes)?;

        println!(
            "{}",
            render_update_report(
                "Update installed",
                Some(VERSION),
                Some(&latest_version),
                Some(&format!(
                    "Installed {} from GitHub release assets for {}.",
                    selected.binary.name,
                    current_target()
                )),
                Some(&release.body),
            )
        );
        Ok(())
    }
}

fn fetch_latest_release() -> Result<Option<GitHubRelease>, Box<dyn std::error::Error>> {
    let client = build_self_update_client()?;
    let response = client
        .get(SELF_UPDATE_LATEST_RELEASE_URL)
        .header(reqwest::header::ACCEPT, "application/vnd.github+json")
        .send()?;

    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }

    let response = response.error_for_status()?;
    Ok(Some(response.json()?))
}

fn build_self_update_client() -> Result<BlockingClient, reqwest::Error> {
    BlockingClient::builder()
        .user_agent(SELF_UPDATE_USER_AGENT)
        .build()
}

fn download_bytes(
    client: &BlockingClient,
    url: &str,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let response = client.get(url).send()?.error_for_status()?;
    Ok(response.bytes()?.to_vec())
}

fn download_text(client: &BlockingClient, url: &str) -> Result<String, Box<dyn std::error::Error>> {
    let response = client.get(url).send()?.error_for_status()?;
    Ok(response.text()?)
}

fn normalize_version_tag(version: &str) -> String {
    version.trim().trim_start_matches('v').to_string()
}

fn is_newer_version(current: &str, latest: &str) -> bool {
    compare_versions(latest, current).is_gt()
}

fn compare_versions(left: &str, right: &str) -> std::cmp::Ordering {
    let left = normalize_version_tag(left);
    let right = normalize_version_tag(right);
    let left_parts = version_components(&left);
    let right_parts = version_components(&right);
    let max_len = left_parts.len().max(right_parts.len());
    for index in 0..max_len {
        let left_part = *left_parts.get(index).unwrap_or(&0);
        let right_part = *right_parts.get(index).unwrap_or(&0);
        match left_part.cmp(&right_part) {
            std::cmp::Ordering::Equal => {}
            ordering => return ordering,
        }
    }
    std::cmp::Ordering::Equal
}

fn version_components(version: &str) -> Vec<u64> {
    version
        .split(['.', '-'])
        .map(|part| {
            part.chars()
                .take_while(char::is_ascii_digit)
                .collect::<String>()
        })
        .filter(|part| !part.is_empty())
        .filter_map(|part| part.parse::<u64>().ok())
        .collect()
}

fn current_target() -> String {
    BUILD_TARGET.map_or_else(default_target_triple, str::to_string)
}

fn default_target_triple() -> String {
    let os = match env::consts::OS {
        "linux" => "unknown-linux-gnu",
        "macos" => "apple-darwin",
        "windows" => "pc-windows-msvc",
        other => other,
    };
    format!("{}-{os}", env::consts::ARCH)
}

fn target_name_candidates() -> Vec<String> {
    let mut candidates = Vec::new();
    if let Some(target) = BUILD_TARGET {
        candidates.push(target.to_string());
    }
    candidates.push(default_target_triple());
    candidates.push(format!("{}-{}", env::consts::ARCH, env::consts::OS));
    candidates.sort();
    candidates.dedup();
    candidates
}

fn release_asset_candidates() -> Vec<String> {
    let mut candidates = target_name_candidates()
        .into_iter()
        .flat_map(|target| {
            let mut names = vec![format!("pebble-{target}")];
            if env::consts::OS == "windows" {
                names.push(format!("pebble-{target}.exe"));
            }
            names
        })
        .collect::<Vec<_>>();
    if env::consts::OS == "windows" {
        candidates.push("pebble.exe".to_string());
    }
    candidates.push("pebble".to_string());
    candidates.sort();
    candidates.dedup();
    candidates
}

fn select_release_assets(release: &GitHubRelease) -> Result<SelectedReleaseAssets, String> {
    let binary = release_asset_candidates()
        .into_iter()
        .find_map(|candidate| {
            release
                .assets
                .iter()
                .find(|asset| asset.name == candidate)
                .cloned()
        })
        .ok_or_else(|| {
            format!(
                "no binary asset matched target {} (expected one of: {})",
                current_target(),
                release_asset_candidates().join(", ")
            )
        })?;

    let checksum = CHECKSUM_ASSET_CANDIDATES
        .iter()
        .find_map(|candidate| {
            release
                .assets
                .iter()
                .find(|asset| asset.name == *candidate)
                .cloned()
        })
        .ok_or_else(|| {
            format!(
                "release did not include a checksum manifest (expected one of: {})",
                CHECKSUM_ASSET_CANDIDATES.join(", ")
            )
        })?;

    Ok(SelectedReleaseAssets { binary, checksum })
}

fn parse_checksum_for_asset(manifest: &str, asset_name: &str) -> Option<String> {
    manifest.lines().find_map(|line| {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return None;
        }
        if let Some((left, right)) = trimmed.split_once(" = ") {
            return left
                .strip_prefix("SHA256 (")
                .and_then(|value| value.strip_suffix(')'))
                .filter(|file| *file == asset_name)
                .map(|_| right.to_ascii_lowercase());
        }
        let mut parts = trimmed.split_whitespace();
        let checksum = parts.next()?;
        let file = parts
            .next_back()
            .or_else(|| parts.next())?
            .trim_start_matches('*');
        (file == asset_name).then(|| checksum.to_ascii_lowercase())
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn replace_current_executable(binary_bytes: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let current = env::current_exe()?;
    replace_executable_at(&current, binary_bytes)
}

fn replace_executable_at(
    current: &Path,
    binary_bytes: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let temp_path = current.with_extension("download");
    let backup_path = current.with_extension("bak");

    if backup_path.exists() {
        fs::remove_file(&backup_path)?;
    }
    fs::write(&temp_path, binary_bytes)?;
    copy_executable_permissions(current, &temp_path)?;

    fs::rename(current, &backup_path)?;
    if let Err(error) = fs::rename(&temp_path, current) {
        let _ = fs::rename(&backup_path, current);
        let _ = fs::remove_file(&temp_path);
        return Err(format!("failed to replace current executable: {error}").into());
    }

    if let Err(error) = fs::remove_file(&backup_path) {
        eprintln!(
            "warning: failed to remove self-update backup {}: {error}",
            backup_path.display()
        );
    }
    Ok(())
}

#[cfg(unix)]
fn copy_executable_permissions(
    source: &Path,
    destination: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::os::unix::fs::PermissionsExt;

    let mode = fs::metadata(source)?.permissions().mode();
    fs::set_permissions(destination, fs::Permissions::from_mode(mode))?;
    Ok(())
}

#[cfg(not(unix))]
fn copy_executable_permissions(
    _source: &Path,
    _destination: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}

fn render_update_report(
    result: &str,
    current_version: Option<&str>,
    latest_version: Option<&str>,
    detail: Option<&str>,
    changelog: Option<&str>,
) -> String {
    let mut report = String::from("Self-update\n");
    let _ = writeln!(report, "  Repository       {SELF_UPDATE_REPOSITORY}");
    let _ = writeln!(report, "  Result           {result}");
    if let Some(current_version) = current_version {
        let _ = writeln!(report, "  Current version  {current_version}");
    }
    if let Some(latest_version) = latest_version {
        let _ = writeln!(report, "  Latest version   {latest_version}");
    }
    if let Some(detail) = detail {
        let _ = writeln!(report, "  Detail           {detail}");
    }
    let trimmed = changelog.map(str::trim).filter(|value| !value.is_empty());
    if let Some(changelog) = trimmed {
        report.push_str("\nChangelog\n");
        report.push_str(changelog);
    }
    report.trim_end().to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DiagnosticLevel {
    Ok,
    Warn,
    Fail,
}

impl DiagnosticLevel {
    const fn label(self) -> &'static str {
        match self {
            Self::Ok => "OK",
            Self::Warn => "WARN",
            Self::Fail => "FAIL",
        }
    }

    const fn is_failure(self) -> bool {
        matches!(self, Self::Fail)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DiagnosticCheck {
    name: &'static str,
    level: DiagnosticLevel,
    summary: String,
    details: Vec<String>,
}

impl DiagnosticCheck {
    fn new(name: &'static str, level: DiagnosticLevel, summary: impl Into<String>) -> Self {
        Self {
            name,
            level,
            summary: summary.into(),
            details: Vec::new(),
        }
    }

    fn with_details(mut self, details: Vec<String>) -> Self {
        self.details = details;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ConfigFileCheck {
    path: PathBuf,
    exists: bool,
    valid: bool,
    note: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DoctorReport {
    checks: Vec<DiagnosticCheck>,
}

impl DoctorReport {
    fn has_failures(&self) -> bool {
        self.checks.iter().any(|check| check.level.is_failure())
    }

    fn render(&self) -> String {
        let mut lines = vec!["Doctor diagnostics".to_string()];
        let ok_count = self
            .checks
            .iter()
            .filter(|check| check.level == DiagnosticLevel::Ok)
            .count();
        let warn_count = self
            .checks
            .iter()
            .filter(|check| check.level == DiagnosticLevel::Warn)
            .count();
        let fail_count = self
            .checks
            .iter()
            .filter(|check| check.level == DiagnosticLevel::Fail)
            .count();
        lines.push(format!(
            "Summary\n  OK               {ok_count}\n  Warnings         {warn_count}\n  Failures         {fail_count}"
        ));
        lines.extend(self.checks.iter().map(render_diagnostic_check));
        lines.join("\n\n")
    }
}

fn render_diagnostic_check(check: &DiagnosticCheck) -> String {
    let mut section = vec![format!(
        "{}\n  Status           {}\n  Summary          {}",
        check.name,
        check.level.label(),
        check.summary
    )];
    if !check.details.is_empty() {
        section.push("  Details".to_string());
        section.extend(check.details.iter().map(|detail| format!("    - {detail}")));
    }
    section.join("\n")
}

fn run_doctor() -> Result<(), Box<dyn std::error::Error>> {
    let cwd = env::current_dir()?;
    let config_loader = ConfigLoader::default_for(&cwd);
    let config = config_loader.load();
    let report = DoctorReport {
        checks: vec![
            check_api_key_validity(),
            check_web_tools_health(),
            check_config_files(&config_loader, config.as_ref()),
            check_git_availability(&cwd),
            check_mcp_server_health(&cwd),
            check_network_connectivity(),
            check_system_info(&cwd, config.as_ref().ok()),
        ],
    };
    println!("{}", report.render());
    if report.has_failures() {
        return Err("doctor found failing checks".into());
    }
    Ok(())
}

fn check_api_key_validity() -> DiagnosticCheck {
    let api_key = match resolve_nanogpt_api_key() {
        Ok(value) => value,
        Err(ApiError::MissingApiKey) => {
            return DiagnosticCheck::new(
                "API key validity",
                DiagnosticLevel::Warn,
                "no NanoGPT API key is configured",
            );
        }
        Err(error) => {
            return DiagnosticCheck::new(
                "API key validity",
                DiagnosticLevel::Fail,
                format!("failed to resolve NanoGPT API key: {error}"),
            );
        }
    };

    let request = MessageRequest {
        model: default_model_or(DEFAULT_MODEL),
        max_tokens: 1,
        messages: vec![InputMessage::user_text("Reply with OK.")],
        system: None,
        tools: None,
        tool_choice: None,
        thinking: None,
        reasoning_effort: None,
        fast_mode: false,
        stream: false,
    };

    let runtime = match tokio::runtime::Runtime::new() {
        Ok(runtime) => runtime,
        Err(error) => {
            return DiagnosticCheck::new(
                "API key validity",
                DiagnosticLevel::Fail,
                format!("failed to create async runtime: {error}"),
            );
        }
    };

    match runtime.block_on(NanoGptClient::new(api_key).send_message(&request)) {
        Ok(response) => DiagnosticCheck::new(
            "API key validity",
            DiagnosticLevel::Ok,
            "NanoGPT API accepted the configured API key",
        )
        .with_details(vec![format!(
            "request_id={} input_tokens={} output_tokens={}",
            response.request_id.unwrap_or_else(|| "<none>".to_string()),
            response.usage.input_tokens,
            response.usage.output_tokens
        )]),
        Err(ApiError::Api { status, .. }) if status == 401 || status == 403 => {
            DiagnosticCheck::new(
                "API key validity",
                DiagnosticLevel::Fail,
                format!("NanoGPT API rejected the API key with HTTP {status}"),
            )
        }
        Err(error) => DiagnosticCheck::new(
            "API key validity",
            DiagnosticLevel::Warn,
            format!("unable to conclusively validate the API key: {error}"),
        ),
    }
}

fn validate_config_file(path: &Path) -> ConfigFileCheck {
    match fs::read_to_string(path) {
        Ok(contents) => {
            if contents.trim().is_empty() {
                return ConfigFileCheck {
                    path: path.to_path_buf(),
                    exists: true,
                    valid: true,
                    note: "exists but is empty".to_string(),
                };
            }
            match serde_json::from_str::<serde_json::Value>(&contents) {
                Ok(serde_json::Value::Object(_)) => ConfigFileCheck {
                    path: path.to_path_buf(),
                    exists: true,
                    valid: true,
                    note: "valid JSON object".to_string(),
                },
                Ok(_) => ConfigFileCheck {
                    path: path.to_path_buf(),
                    exists: true,
                    valid: false,
                    note: "top-level JSON value is not an object".to_string(),
                },
                Err(error) => ConfigFileCheck {
                    path: path.to_path_buf(),
                    exists: true,
                    valid: false,
                    note: format!("invalid JSON: {error}"),
                },
            }
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => ConfigFileCheck {
            path: path.to_path_buf(),
            exists: false,
            valid: true,
            note: "not present".to_string(),
        },
        Err(error) => ConfigFileCheck {
            path: path.to_path_buf(),
            exists: true,
            valid: false,
            note: format!("unreadable: {error}"),
        },
    }
}

fn check_config_files(
    config_loader: &ConfigLoader,
    config: Result<&runtime::RuntimeConfig, &runtime::ConfigError>,
) -> DiagnosticCheck {
    let file_checks = config_loader
        .discover()
        .into_iter()
        .map(|entry| validate_config_file(&entry.path))
        .collect::<Vec<_>>();
    let existing_count = file_checks.iter().filter(|check| check.exists).count();
    let invalid_count = file_checks
        .iter()
        .filter(|check| check.exists && !check.valid)
        .count();
    let mut details = file_checks
        .iter()
        .map(|check| format!("{} => {}", check.path.display(), check.note))
        .collect::<Vec<_>>();
    match config {
        Ok(runtime_config) => details.push(format!(
            "merged load succeeded with {} loaded file(s)",
            runtime_config.loaded_entries().len()
        )),
        Err(error) => details.push(format!("merged load failed: {error}")),
    }
    DiagnosticCheck::new(
        "Config files",
        if invalid_count > 0 || config.is_err() {
            DiagnosticLevel::Fail
        } else if existing_count == 0 {
            DiagnosticLevel::Warn
        } else {
            DiagnosticLevel::Ok
        },
        format!(
            "discovered {} candidate file(s), {} existing, {} invalid",
            file_checks.len(),
            existing_count,
            invalid_count
        ),
    )
    .with_details(details)
}

fn check_git_availability(cwd: &Path) -> DiagnosticCheck {
    match Command::new("git").arg("--version").output() {
        Ok(version_output) if version_output.status.success() => {
            let version = String::from_utf8_lossy(&version_output.stdout)
                .trim()
                .to_string();
            match Command::new("git")
                .args(["rev-parse", "--show-toplevel"])
                .current_dir(cwd)
                .output()
            {
                Ok(root_output) if root_output.status.success() => DiagnosticCheck::new(
                    "Git availability",
                    DiagnosticLevel::Ok,
                    "git is installed and the current directory is inside a repository",
                )
                .with_details(vec![
                    version,
                    format!(
                        "repo_root={}",
                        String::from_utf8_lossy(&root_output.stdout).trim()
                    ),
                ]),
                Ok(_) => DiagnosticCheck::new(
                    "Git availability",
                    DiagnosticLevel::Warn,
                    "git is installed but the current directory is not a repository",
                )
                .with_details(vec![version]),
                Err(error) => DiagnosticCheck::new(
                    "Git availability",
                    DiagnosticLevel::Warn,
                    format!("git is installed but repo detection failed: {error}"),
                )
                .with_details(vec![version]),
            }
        }
        Ok(output) => DiagnosticCheck::new(
            "Git availability",
            DiagnosticLevel::Fail,
            format!("git --version exited with status {}", output.status),
        ),
        Err(error) => DiagnosticCheck::new(
            "Git availability",
            DiagnosticLevel::Fail,
            format!("failed to execute git: {error}"),
        ),
    }
}

fn check_mcp_server_health(cwd: &Path) -> DiagnosticCheck {
    match load_mcp_catalog(cwd) {
        Ok(catalog) if catalog.servers.is_empty() => DiagnosticCheck::new(
            "MCP server health",
            DiagnosticLevel::Warn,
            "no MCP servers are configured",
        ),
        Ok(catalog) => {
            let level = if catalog.servers.iter().any(|server| !server.loaded) {
                DiagnosticLevel::Warn
            } else {
                DiagnosticLevel::Ok
            };
            DiagnosticCheck::new(
                "MCP server health",
                level,
                format!("checked {} configured MCP server(s)", catalog.servers.len()),
            )
            .with_details(
                catalog
                    .servers
                    .iter()
                    .map(|server| {
                        format!(
                            "{} [{:?}] {} tool(s): {}",
                            server.server_name, server.transport, server.tool_count, server.note
                        )
                    })
                    .collect(),
            )
        }
        Err(error) => DiagnosticCheck::new(
            "MCP server health",
            DiagnosticLevel::Fail,
            format!("failed to inspect MCP servers: {error}"),
        ),
    }
}

fn check_network_connectivity() -> DiagnosticCheck {
    let address = match ("nano-gpt.com", 443).to_socket_addrs() {
        Ok(mut addrs) => match addrs.next() {
            Some(addr) => addr,
            None => {
                return DiagnosticCheck::new(
                    "Network connectivity",
                    DiagnosticLevel::Fail,
                    "DNS resolution returned no addresses for nano-gpt.com",
                );
            }
        },
        Err(error) => {
            return DiagnosticCheck::new(
                "Network connectivity",
                DiagnosticLevel::Fail,
                format!("failed to resolve nano-gpt.com: {error}"),
            );
        }
    };
    match TcpStream::connect_timeout(&address, Duration::from_secs(5)) {
        Ok(stream) => {
            let _ = stream.shutdown(std::net::Shutdown::Both);
            DiagnosticCheck::new(
                "Network connectivity",
                DiagnosticLevel::Ok,
                format!("connected to {address}"),
            )
        }
        Err(error) => DiagnosticCheck::new(
            "Network connectivity",
            DiagnosticLevel::Fail,
            format!("failed to connect to {address}: {error}"),
        ),
    }
}

fn check_web_tools_health() -> DiagnosticCheck {
    let service = current_service_or_default();
    let base_url = resolve_base_url_for(service);
    let api_key_configured = resolve_api_key_for(service).is_ok();
    match current_tool_registry() {
        Ok(registry) => {
            let mut has_search = false;
            let mut has_scrape = false;
            for entry in registry.entries() {
                if entry.definition.name == "WebSearch" {
                    has_search = true;
                }
                if entry.definition.name == "WebScrape" {
                    has_scrape = true;
                }
            }
            let level = if has_search && has_scrape && api_key_configured {
                DiagnosticLevel::Ok
            } else if has_search || has_scrape {
                DiagnosticLevel::Warn
            } else {
                DiagnosticLevel::Fail
            };
            DiagnosticCheck::new(
                "Web tools",
                level,
                if has_search && has_scrape {
                    "web tools are registered"
                } else {
                    "one or more web tools are unavailable"
                },
            )
            .with_details(vec![
                format!("service={}", service.display_name()),
                format!("base_url={base_url}"),
                format!(
                    "api_key={}",
                    if api_key_configured {
                        "configured"
                    } else {
                        "missing"
                    }
                ),
                format!(
                    "WebSearch={}",
                    if has_search { "available" } else { "missing" }
                ),
                format!(
                    "WebScrape={}",
                    if has_scrape { "available" } else { "missing" }
                ),
            ])
        }
        Err(error) => DiagnosticCheck::new(
            "Web tools",
            DiagnosticLevel::Fail,
            format!("failed to load tool registry: {error}"),
        )
        .with_details(vec![format!("base_url={base_url}")]),
    }
}

fn check_system_info(cwd: &Path, config: Option<&runtime::RuntimeConfig>) -> DiagnosticCheck {
    let mut details = vec![
        format!("os={} arch={}", env::consts::OS, env::consts::ARCH),
        format!("cwd={}", cwd.display()),
        format!("cli_version={VERSION}"),
    ];
    if let Some(config) = config {
        let sandbox_status = resolve_sandbox_status(config.sandbox(), cwd);
        details.push(format!(
            "resolved_model={} loaded_config_files={}",
            config.model().unwrap_or(DEFAULT_MODEL),
            config.loaded_entries().len()
        ));
        details.push(format!(
            "sandbox enabled={} active={} namespace_active={} network_active={} filesystem_mode={}",
            sandbox_status.enabled,
            sandbox_status.active,
            sandbox_status.namespace_active,
            sandbox_status.network_active,
            sandbox_status.filesystem_mode.as_str()
        ));
        if !sandbox_status.allowed_mounts.is_empty() {
            details.push(format!(
                "sandbox_allowed_mounts={}",
                sandbox_status.allowed_mounts.join(", ")
            ));
        }
        if let Some(reason) = sandbox_status.fallback_reason {
            details.push(format!("sandbox_fallback={reason}"));
        }
    }
    DiagnosticCheck::new(
        "System info",
        DiagnosticLevel::Ok,
        "captured local runtime and build metadata",
    )
    .with_details(details)
}

#[cfg(test)]
mod tests {
    use super::{
        append_proxy_text_events, available_runtime_tool_specs, build_system_prompt,
        extract_first_json_object, filter_runtime_tool_specs, format_status_report,
        format_web_tools_status, login_model_guidance, normalize_generated_pebble_md, parse_args,
        parse_auth_command, parse_checksum_for_asset, parse_logout_command, parse_mcp_command,
        parse_model_command, parse_permissions_command, parse_provider_command,
        parse_proxy_command, parse_tool_input_value, prompt_to_content_blocks,
        proxy_response_to_events, push_output_block, remove_saved_credentials,
        render_streamed_tool_call_start, render_tool_result_block, render_update_report,
        resolve_model_alias, response_to_events, should_ignore_stale_secret_submit,
        should_retry_proxy_tool_prompt, strip_markdown_code_fence, trim_trailing_line_endings,
        tuned_tool_description, AssistantEvent, AuthService, CliAction, CliOutputFormat,
        CollaborationMode, CredentialRemovalOutcome, FastMode, GitHubRelease, GitHubReleaseAsset,
        LoginCommand, LogoutCommand, McpCatalog, McpCommand, PebbleRuntimeClient, RuntimeToolSpec,
        StatusContext, StatusUsage, DEFAULT_MAX_TOKENS, DEFAULT_MODEL,
    };
    use crate::proxy::ProxyCommand;
    use api::{ApiService, InputContentBlock, MessageResponse, OutputContentBlock, Usage};
    use runtime::{ContentBlock, ConversationMessage, MessageRole, PermissionMode, TokenUsage};
    use serde_json::json;
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, OnceLock};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};
    use tools::current_tool_registry;

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env test lock should not be poisoned")
    }

    fn with_isolated_config_home<T>(run: impl FnOnce() -> T) -> T {
        let _guard = env_lock();
        let root = std::env::temp_dir().join(format!(
            "pebble-main-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time should be after epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).expect("config dir should exist");
        std::env::set_var("PEBBLE_CONFIG_HOME", &root);
        let output = run();
        std::env::remove_var("PEBBLE_CONFIG_HOME");
        std::fs::remove_dir_all(root).expect("temp config dir should be removed");
        output
    }

    fn tool_specs() -> Vec<RuntimeToolSpec> {
        available_runtime_tool_specs(
            &current_tool_registry().expect("tool registry should load"),
            &McpCatalog::default(),
        )
    }

    #[test]
    fn defaults_to_repl_when_no_args() {
        with_isolated_config_home(|| {
            assert_eq!(
                parse_args(&[]).expect("args should parse"),
                CliAction::Repl {
                    model: DEFAULT_MODEL.to_string(),
                    allowed_tools: None,
                    permission_mode: PermissionMode::WorkspaceWrite,
                    collaboration_mode: CollaborationMode::Build,
                    reasoning_effort: None,
                    fast_mode: FastMode::Off,
                }
            );
        });
    }

    #[test]
    fn stale_empty_secret_submit_is_ignored_briefly() {
        assert!(should_ignore_stale_secret_submit(
            "",
            Duration::from_millis(25)
        ));
        assert!(!should_ignore_stale_secret_submit(
            "sk-live",
            Duration::from_millis(25)
        ));
        assert!(!should_ignore_stale_secret_submit(
            "",
            Duration::from_millis(300)
        ));
    }

    #[test]
    fn trims_trailing_line_endings_from_pasted_secret() {
        assert_eq!(trim_trailing_line_endings("abc123\r\n"), "abc123");
        assert_eq!(trim_trailing_line_endings("abc123\n"), "abc123");
        assert_eq!(trim_trailing_line_endings("abc123"), "abc123");
    }

    #[test]
    fn runtime_client_constructor_defers_api_key_lookup() {
        with_isolated_config_home(|| {
            let original = std::env::var("NANOGPT_API_KEY").ok();
            std::env::remove_var("NANOGPT_API_KEY");

            let client = PebbleRuntimeClient::new(
                ApiService::NanoGpt,
                DEFAULT_MODEL.to_string(),
                DEFAULT_MAX_TOKENS,
                None,
                true,
                false,
                Vec::new(),
                CollaborationMode::Build,
                None,
                FastMode::Off,
                false,
            );

            match original {
                Some(value) => std::env::set_var("NANOGPT_API_KEY", value),
                None => std::env::remove_var("NANOGPT_API_KEY"),
            }

            assert!(
                client.is_ok(),
                "runtime client should initialize without credentials"
            );
        });
    }

    #[test]
    fn synthetic_login_guidance_explains_active_model_mismatch() {
        with_isolated_config_home(|| {
            let note = login_model_guidance(AuthService::Synthetic)
                .expect("synthetic login should show model guidance by default");
            assert!(note.contains("current model is `zai-org/glm-5.1`"));
            assert!(note.contains("Logging into Synthetic saves credentials"));
            assert!(note.contains("prefixed with `hf:`"));
        });
    }

    #[test]
    fn tuned_web_tool_descriptions_push_search_and_scrape_workflow() {
        let search = tuned_tool_description("WebSearch", "Search the web.");
        let scrape = tuned_tool_description("WebScrape", "Scrape pages.");
        let fetch = tuned_tool_description("WebFetch", "Fetch a URL.");

        assert!(search.contains("current information"));
        assert!(scrape.contains("readable page content"));
        assert!(fetch.contains("WebScrape"));
    }

    #[test]
    fn system_prompt_includes_web_research_guidance() {
        let prompt =
            build_system_prompt(ApiService::NanoGpt, DEFAULT_MODEL, CollaborationMode::Build)
                .expect("system prompt should build")
                .join("\n\n");
        assert!(prompt.contains("# Web Research Guidance"));
        assert!(prompt.contains("WebSearch"));
        assert!(prompt.contains("WebScrape"));
    }

    #[test]
    fn web_tools_status_mentions_auth_and_tool_availability() {
        let summary = format_web_tools_status();
        assert!(summary.contains("api_key="));
        assert!(summary.contains("web_search="));
        assert!(summary.contains("web_scrape="));
    }

    #[test]
    fn status_report_includes_web_tools_section() {
        let report = format_status_report(
            ApiService::NanoGpt,
            DEFAULT_MODEL,
            StatusUsage {
                message_count: 0,
                turns: 0,
                latest: TokenUsage {
                    input_tokens: 0,
                    output_tokens: 0,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 0,
                },
                cumulative: TokenUsage {
                    input_tokens: 0,
                    output_tokens: 0,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 0,
                },
                estimated_tokens: 0,
            },
            "workspace-write",
            Some("<platform default>"),
            false,
            CollaborationMode::Build,
            None,
            FastMode::Off,
            &McpCatalog::default(),
            &StatusContext {
                cwd: PathBuf::from("."),
                session_path: None,
                loaded_config_files: 0,
                discovered_config_files: 0,
                instruction_file_count: 0,
                memory_file_count: 0,
                project_root: None,
                git_branch: None,
                sandbox_summary: "sandbox".to_string(),
                web_tools_summary: "web".to_string(),
            },
        );
        assert!(report.contains("Web Tools"));
    }

    #[test]
    fn parses_prompt_subcommand() {
        with_isolated_config_home(|| {
            let args = vec![
                "prompt".to_string(),
                "hello".to_string(),
                "world".to_string(),
            ];
            assert_eq!(
                parse_args(&args).expect("args should parse"),
                CliAction::Prompt {
                    prompt: "hello world".to_string(),
                    model: DEFAULT_MODEL.to_string(),
                    allowed_tools: None,
                    permission_mode: PermissionMode::WorkspaceWrite,
                    collaboration_mode: CollaborationMode::Build,
                    reasoning_effort: None,
                    fast_mode: FastMode::Off,
                    output_format: CliOutputFormat::Text,
                }
            );
        });
    }

    #[test]
    fn parses_bare_prompt_with_json_output_flag() {
        with_isolated_config_home(|| {
            let args = vec![
                "--output-format=json".to_string(),
                "summarize".to_string(),
                "this".to_string(),
                "repo".to_string(),
            ];
            assert_eq!(
                parse_args(&args).expect("args should parse"),
                CliAction::Prompt {
                    prompt: "summarize this repo".to_string(),
                    model: DEFAULT_MODEL.to_string(),
                    allowed_tools: None,
                    permission_mode: PermissionMode::WorkspaceWrite,
                    collaboration_mode: CollaborationMode::Build,
                    reasoning_effort: None,
                    fast_mode: FastMode::Off,
                    output_format: CliOutputFormat::Json,
                }
            );
        });
    }

    #[test]
    fn parses_dash_p_prompt_shorthand() {
        with_isolated_config_home(|| {
            let args = vec![
                "-p".to_string(),
                "summarize".to_string(),
                "this".to_string(),
            ];
            assert_eq!(
                parse_args(&args).expect("args should parse"),
                CliAction::Prompt {
                    prompt: "summarize this".to_string(),
                    model: DEFAULT_MODEL.to_string(),
                    allowed_tools: None,
                    permission_mode: PermissionMode::WorkspaceWrite,
                    collaboration_mode: CollaborationMode::Build,
                    reasoning_effort: None,
                    fast_mode: FastMode::Off,
                    output_format: CliOutputFormat::Text,
                }
            );
        });
    }

    #[test]
    fn parses_print_flag_as_text_output() {
        with_isolated_config_home(|| {
            let args = vec![
                "--output-format=json".to_string(),
                "--print".to_string(),
                "summarize".to_string(),
                "this".to_string(),
            ];
            assert_eq!(
                parse_args(&args).expect("args should parse"),
                CliAction::Prompt {
                    prompt: "summarize this".to_string(),
                    model: DEFAULT_MODEL.to_string(),
                    allowed_tools: None,
                    permission_mode: PermissionMode::WorkspaceWrite,
                    collaboration_mode: CollaborationMode::Build,
                    reasoning_effort: None,
                    fast_mode: FastMode::Off,
                    output_format: CliOutputFormat::Text,
                }
            );
        });
    }

    #[test]
    fn parses_allowed_tools_flags_with_aliases_and_lists() {
        with_isolated_config_home(|| {
            let args = vec![
                "--allowedTools".to_string(),
                "read,glob".to_string(),
                "--allowed-tools=write_file".to_string(),
            ];
            assert_eq!(
                parse_args(&args).expect("args should parse"),
                CliAction::Repl {
                    model: DEFAULT_MODEL.to_string(),
                    allowed_tools: Some(
                        ["glob_search", "read_file", "write_file"]
                            .into_iter()
                            .map(str::to_string)
                            .collect()
                    ),
                    permission_mode: PermissionMode::WorkspaceWrite,
                    collaboration_mode: CollaborationMode::Build,
                    reasoning_effort: None,
                    fast_mode: FastMode::Off,
                }
            );
        });
    }

    #[test]
    fn parses_permission_mode_flag() {
        with_isolated_config_home(|| {
            let args = vec!["--permission-mode=read-only".to_string()];
            assert_eq!(
                parse_args(&args).expect("args should parse"),
                CliAction::Repl {
                    model: DEFAULT_MODEL.to_string(),
                    allowed_tools: None,
                    permission_mode: PermissionMode::ReadOnly,
                    collaboration_mode: CollaborationMode::Build,
                    reasoning_effort: None,
                    fast_mode: FastMode::Off,
                }
            );
        });
    }

    #[test]
    fn rejects_unknown_allowed_tools() {
        with_isolated_config_home(|| {
            let error = parse_args(&["--allowedTools".to_string(), "teleport".to_string()])
                .expect_err("tool should be rejected");
            assert!(error.contains("unsupported tool in --allowedTools: teleport"));
        });
    }

    #[test]
    fn parses_login_subcommand() {
        with_isolated_config_home(|| {
            let args = vec!["login".to_string(), "--api-key=nano-key".to_string()];
            assert_eq!(
                parse_args(&args).expect("args should parse"),
                CliAction::Login {
                    service: None,
                    api_key: Some("nano-key".to_string()),
                }
            );
        });
    }

    #[test]
    fn parses_logout_subcommand() {
        with_isolated_config_home(|| {
            let args = vec!["logout".to_string(), "openai-codex".to_string()];
            assert_eq!(
                parse_args(&args).expect("args should parse"),
                CliAction::Logout {
                    service: Some(AuthService::OpenAiCodex),
                }
            );
        });
    }

    #[test]
    fn parses_model_subcommand() {
        with_isolated_config_home(|| {
            let args = vec!["model".to_string(), "openai/gpt-5.2".to_string()];
            assert_eq!(
                parse_args(&args).expect("args should parse"),
                CliAction::Model {
                    model: Some("openai/gpt-5.2".to_string()),
                }
            );
        });
    }

    #[test]
    fn parses_provider_subcommand() {
        with_isolated_config_home(|| {
            let args = vec!["provider".to_string(), "openrouter".to_string()];
            assert_eq!(
                parse_args(&args).expect("args should parse"),
                CliAction::Provider {
                    provider: Some("openrouter".to_string()),
                }
            );
        });
    }

    #[test]
    fn parses_proxy_subcommand() {
        with_isolated_config_home(|| {
            let args = vec!["proxy".to_string(), "on".to_string()];
            assert_eq!(
                parse_args(&args).expect("args should parse"),
                CliAction::Proxy {
                    mode: ProxyCommand::Enable,
                }
            );
        });
    }

    #[test]
    fn parses_mcp_subcommand() {
        with_isolated_config_home(|| {
            let args = vec!["mcp".to_string(), "tools".to_string()];
            assert_eq!(
                parse_args(&args).expect("args should parse"),
                CliAction::Mcp {
                    action: McpCommand::Tools,
                }
            );
            let args = vec![
                "mcp".to_string(),
                "disable".to_string(),
                "context7".to_string(),
            ];
            assert_eq!(
                parse_args(&args).expect("args should parse"),
                CliAction::Mcp {
                    action: McpCommand::Disable {
                        name: "context7".to_string(),
                    },
                }
            );
        });
    }

    #[test]
    fn parses_system_prompt_options() {
        with_isolated_config_home(|| {
            let args = vec![
                "system-prompt".to_string(),
                "--cwd".to_string(),
                "/tmp/project".to_string(),
                "--date".to_string(),
                "2026-04-01".to_string(),
            ];
            assert_eq!(
                parse_args(&args).expect("args should parse"),
                CliAction::PrintSystemPrompt {
                    cwd: PathBuf::from("/tmp/project"),
                    date: "2026-04-01".to_string(),
                }
            );
        });
    }

    #[test]
    fn parses_resume_flag_with_slash_command() {
        with_isolated_config_home(|| {
            let args = vec![
                "--resume".to_string(),
                "session.json".to_string(),
                "/compact".to_string(),
            ];
            assert_eq!(
                parse_args(&args).expect("args should parse"),
                CliAction::ResumeSession {
                    session_path: Some(PathBuf::from("session.json")),
                    commands: vec!["/compact".to_string()],
                }
            );
        });
    }

    #[test]
    fn parses_resume_flag_with_multiple_slash_commands() {
        with_isolated_config_home(|| {
            let args = vec![
                "--resume".to_string(),
                "session.json".to_string(),
                "/status".to_string(),
                "/export".to_string(),
            ];
            assert_eq!(
                parse_args(&args).expect("args should parse"),
                CliAction::ResumeSession {
                    session_path: Some(PathBuf::from("session.json")),
                    commands: vec!["/status".to_string(), "/export".to_string()],
                }
            );
        });
    }

    #[test]
    fn parses_resume_without_path_as_picker_action() {
        with_isolated_config_home(|| {
            let args = vec!["resume".to_string()];
            assert_eq!(
                parse_args(&args).expect("args should parse"),
                CliAction::ResumeSession {
                    session_path: None,
                    commands: Vec::new(),
                }
            );
        });
    }

    #[test]
    fn parses_version_flag() {
        with_isolated_config_home(|| {
            let args = vec!["--version".to_string()];
            assert_eq!(
                parse_args(&args).expect("args should parse"),
                CliAction::Version
            );
        });
    }

    #[test]
    fn parses_self_update_subcommand() {
        with_isolated_config_home(|| {
            assert_eq!(
                parse_args(&["self-update".to_string()]).expect("self-update should parse"),
                CliAction::SelfUpdate
            );
        });
    }

    #[test]
    fn parses_checksum_manifest_for_named_asset() {
        let manifest = "abc123 *pebble-aarch64-apple-darwin\ndef456 other-file\n";
        assert_eq!(
            parse_checksum_for_asset(manifest, "pebble-aarch64-apple-darwin"),
            Some("abc123".to_string())
        );
    }

    #[test]
    fn select_release_assets_requires_checksum_file() {
        let asset_name = super::release_asset_candidates()
            .into_iter()
            .next()
            .expect("at least one asset candidate");
        let release = GitHubRelease {
            tag_name: "v0.2.0".to_string(),
            body: String::new(),
            assets: vec![GitHubReleaseAsset {
                name: asset_name,
                browser_download_url: "https://example.invalid/pebble".to_string(),
            }],
        };

        let error =
            super::select_release_assets(&release).expect_err("missing checksum should error");
        assert!(error.contains("checksum manifest"));
    }

    #[test]
    fn update_report_includes_changelog_when_present() {
        let report = render_update_report(
            "Already up to date",
            Some("0.1.3"),
            Some("0.1.3"),
            Some("No action taken."),
            Some("- Added self-update"),
        );
        assert!(report.contains("Self-update"));
        assert!(report.contains("Changelog"));
        assert!(report.contains("- Added self-update"));
        assert!(report.contains("nanogpt-community/pebble"));
    }

    #[test]
    fn filtered_tool_specs_respect_allowlist() {
        let allowed = ["read_file", "grep_search"]
            .into_iter()
            .map(str::to_string)
            .collect();
        let filtered = filter_runtime_tool_specs(
            available_runtime_tool_specs(
                &current_tool_registry().expect("tool registry should load"),
                &McpCatalog::default(),
            ),
            Some(&allowed),
        );
        let names = filtered
            .into_iter()
            .map(|spec| spec.name)
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["read_file", "grep_search"]);
    }

    #[test]
    fn parses_auth_slash_command() {
        assert_eq!(
            parse_auth_command("/login"),
            Some(LoginCommand {
                service: None,
                api_key: None,
            })
        );
        assert_eq!(
            parse_auth_command("/auth nano-key"),
            Some(LoginCommand {
                service: None,
                api_key: Some("nano-key".to_string()),
            })
        );
        assert_eq!(
            parse_auth_command("/login synthetic"),
            Some(LoginCommand {
                service: Some(AuthService::Synthetic),
                api_key: None,
            })
        );
        assert_eq!(
            parse_auth_command("/login openai-codex"),
            Some(LoginCommand {
                service: Some(AuthService::OpenAiCodex),
                api_key: None,
            })
        );
        assert_eq!(
            parse_auth_command("/login opencode-go"),
            Some(LoginCommand {
                service: Some(AuthService::OpencodeGo),
                api_key: None,
            })
        );
        assert_eq!(
            parse_auth_command("/login exa"),
            Some(LoginCommand {
                service: Some(AuthService::Exa),
                api_key: None,
            })
        );
        assert_eq!(parse_auth_command("/status"), None);
    }

    #[test]
    fn parses_logout_slash_command() {
        assert_eq!(
            parse_logout_command("/logout"),
            Some(LogoutCommand { service: None })
        );
        assert_eq!(
            parse_logout_command("/logout openai-codex"),
            Some(LogoutCommand {
                service: Some(AuthService::OpenAiCodex),
            })
        );
        assert_eq!(parse_logout_command("/status"), None);
    }

    #[test]
    fn removes_saved_credentials_for_selected_service() {
        with_isolated_config_home(|| {
            let config_home =
                std::env::var("PEBBLE_CONFIG_HOME").expect("isolated config home should be set");
            let credentials_path = PathBuf::from(config_home).join("credentials.json");
            std::fs::write(
                &credentials_path,
                serde_json::json!({
                    "openai_codex_auth": {
                        "access_token": "token",
                        "refresh_token": "refresh"
                    },
                    "nanogpt_api_key": "nano-key"
                })
                .to_string(),
            )
            .expect("credentials should be written");

            let outcome = remove_saved_credentials(AuthService::OpenAiCodex)
                .expect("logout should remove saved credentials");
            assert_eq!(
                outcome,
                CredentialRemovalOutcome::Removed {
                    path: credentials_path.clone(),
                }
            );

            let parsed: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(&credentials_path)
                    .expect("credentials should remain readable"),
            )
            .expect("credentials should remain valid json");
            assert!(parsed.get("openai_codex_auth").is_none());
            assert_eq!(
                parsed
                    .get("nanogpt_api_key")
                    .and_then(serde_json::Value::as_str),
                Some("nano-key")
            );
        });
    }

    #[test]
    fn parses_bypass_as_danger_full_access() {
        assert!(matches!(
            parse_permissions_command("/bypass"),
            Some(Ok(Some(PermissionMode::DangerFullAccess)))
        ));
        assert!(matches!(
            parse_permissions_command("/bypass now"),
            Some(Err(message)) if message == "/bypass does not accept arguments"
        ));
    }

    #[test]
    fn parses_model_slash_command() {
        assert_eq!(parse_model_command("/model"), Some(None));
        assert_eq!(
            parse_model_command("/models openai/gpt-5.2"),
            Some(Some("openai/gpt-5.2".to_string()))
        );
        assert_eq!(parse_model_command("/status"), None);
    }

    #[test]
    fn resolves_known_pebble_model_aliases() {
        assert_eq!(resolve_model_alias("default"), "zai-org/glm-5.1");
        assert_eq!(resolve_model_alias("glm"), "zai-org/glm-5.1");
        assert_eq!(resolve_model_alias("glm5"), "zai-org/glm-5");
        assert_eq!(resolve_model_alias("glm-5.1"), "zai-org/glm-5.1");
        assert_eq!(resolve_model_alias("openai/gpt-5.2"), "openai/gpt-5.2");
    }

    #[test]
    fn parses_provider_slash_command() {
        assert_eq!(parse_provider_command("/provider"), Some(None));
        assert_eq!(
            parse_provider_command("/providers openrouter"),
            Some(Some("openrouter".to_string()))
        );
        assert_eq!(parse_provider_command("/status"), None);
    }

    #[test]
    fn parses_proxy_slash_command() {
        assert_eq!(
            parse_proxy_command("/proxy").expect("proxy command should parse"),
            Ok(ProxyCommand::Toggle)
        );
        assert_eq!(
            parse_proxy_command("/proxy status").expect("proxy status should parse"),
            Ok(ProxyCommand::Status)
        );
        assert!(parse_proxy_command("/status").is_none());
    }

    #[test]
    fn parses_mcp_slash_command() {
        assert_eq!(
            parse_mcp_command("/mcp tools").expect("mcp command should parse"),
            Ok(McpCommand::Tools)
        );
        assert_eq!(
            parse_mcp_command("/mcp enable context7").expect("mcp enable should parse"),
            Ok(McpCommand::Enable {
                name: "context7".to_string(),
            })
        );
        assert_eq!(
            parse_mcp_command("/mcp").expect("mcp default should parse"),
            Ok(McpCommand::Status)
        );
        assert!(parse_mcp_command("/status").is_none());
    }

    #[test]
    fn recovers_json_object_from_noisy_proxy_input() {
        let recovered = extract_first_json_object(
            "Here you go {\"path\":\"README.md\",\"offset\":0} and then some commentary",
        )
        .expect("json object should be recovered");
        assert_eq!(
            recovered,
            serde_json::json!({"path":"README.md","offset":0})
        );
    }

    #[test]
    fn parses_inline_proxy_arg_fragment_for_tool_execution() {
        let value = parse_tool_input_value(
            "read_file",
            "<arg name=\"path\">README.md</arg><arg name=\"offset\" type=\"integer\">0</arg>",
            &tool_specs(),
        )
        .expect("proxy arg fragment should parse");
        assert_eq!(value, serde_json::json!({"path":"README.md","offset":0}));
    }

    #[test]
    fn converts_tool_roundtrip_messages() {
        let messages = vec![
            ConversationMessage::user_text("hello"),
            ConversationMessage::assistant(vec![ContentBlock::ToolUse {
                id: "tool-1".to_string(),
                name: "bash".to_string(),
                input: "{\"command\":\"pwd\"}".to_string(),
            }]),
            ConversationMessage {
                role: MessageRole::Tool,
                blocks: vec![ContentBlock::ToolResult {
                    tool_use_id: "tool-1".to_string(),
                    tool_name: "bash".to_string(),
                    output: "ok".to_string(),
                    is_error: false,
                }],
                usage: None,
            },
        ];

        let converted = super::convert_messages(&messages, ApiService::NanoGpt)
            .expect("messages should convert");
        assert_eq!(converted.len(), 3);
        assert_eq!(converted[1].role, "assistant");
        assert_eq!(converted[2].role, "user");
    }

    #[test]
    fn convert_messages_drops_dangling_tool_use_blocks() {
        let messages = vec![
            ConversationMessage::user_text("hello"),
            ConversationMessage::assistant(vec![ContentBlock::ToolUse {
                id: "tool-1".to_string(),
                name: "bash".to_string(),
                input: "{\"command\":\"pwd\"}".to_string(),
            }]),
            ConversationMessage::assistant(vec![ContentBlock::Text {
                text: "I can still answer normally.".to_string(),
            }]),
        ];

        let converted = super::convert_messages(&messages, ApiService::NanoGpt)
            .expect("messages should convert");
        assert_eq!(converted.len(), 2);
        assert_eq!(converted[0].role, "user");
        assert_eq!(converted[1].role, "assistant");
        assert!(matches!(
            &converted[1].content[0],
            InputContentBlock::Text { text } if text == "I can still answer normally."
        ));
    }

    #[test]
    fn convert_messages_drops_orphan_tool_results() {
        let messages = vec![
            ConversationMessage::user_text("hello"),
            ConversationMessage::tool_result("tool-missing", "bash", "ok", false),
        ];

        let converted = super::convert_messages(&messages, ApiService::NanoGpt)
            .expect("messages should convert");
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "user");
        assert!(matches!(
            &converted[0].content[0],
            InputContentBlock::Text { text } if text == "hello"
        ));
    }

    #[test]
    fn convert_messages_preserves_reasoning_for_opencode_go_assistant_messages() {
        let messages = vec![
            ConversationMessage::assistant(vec![
                ContentBlock::Thinking {
                    text: "reasoning trail".to_string(),
                    signature: None,
                },
                ContentBlock::ToolUse {
                    id: "tool-1".to_string(),
                    name: "bash".to_string(),
                    input: "{\"command\":\"pwd\"}".to_string(),
                },
            ]),
            ConversationMessage::tool_result("tool-1", "bash", "ok", false),
        ];

        let converted = super::convert_messages(&messages, ApiService::OpencodeGo)
            .expect("messages should convert");
        assert_eq!(converted.len(), 2);
        assert_eq!(
            converted[0].reasoning_content.as_deref(),
            Some("reasoning trail")
        );
        assert_eq!(converted[0].reasoning.as_deref(), Some("reasoning trail"));
    }

    #[test]
    fn prompt_to_content_blocks_keeps_text_only_prompt() {
        let blocks = prompt_to_content_blocks("hello world", Path::new("."))
            .expect("text prompt should parse");
        assert_eq!(
            blocks,
            vec![InputContentBlock::Text {
                text: "hello world".to_string()
            }]
        );
    }

    #[test]
    fn prompt_to_content_blocks_embeds_at_image_refs() {
        let temp = temp_fixture_dir("at-image-ref");
        let image_path = temp.join("sample.png");
        std::fs::write(&image_path, [1_u8, 2, 3]).expect("fixture write");
        let prompt = format!("describe @{} please", image_path.display());

        let blocks =
            prompt_to_content_blocks(&prompt, Path::new(".")).expect("image ref should parse");

        assert!(matches!(
            &blocks[0],
            InputContentBlock::Text { text } if text == "describe "
        ));
        assert!(matches!(
            &blocks[1],
            InputContentBlock::Image { source }
                if source.kind == "base64"
                    && source.media_type == "image/png"
                    && source.data == "AQID"
        ));
        assert!(matches!(
            &blocks[2],
            InputContentBlock::Text { text } if text == " please"
        ));
    }

    #[test]
    fn prompt_to_content_blocks_embeds_markdown_image_refs() {
        let temp = temp_fixture_dir("markdown-image-ref");
        let image_path = temp.join("sample.webp");
        std::fs::write(&image_path, [255_u8]).expect("fixture write");
        let prompt = format!("see ![asset]({}) now", image_path.display());

        let blocks = prompt_to_content_blocks(&prompt, Path::new("."))
            .expect("markdown image ref should parse");

        assert!(matches!(
            &blocks[1],
            InputContentBlock::Image { source }
                if source.media_type == "image/webp" && source.data == "/w=="
        ));
    }

    #[test]
    fn prompt_to_content_blocks_rejects_unsupported_formats() {
        let temp = temp_fixture_dir("unsupported-image-ref");
        let image_path = temp.join("sample.bmp");
        std::fs::write(&image_path, [1_u8]).expect("fixture write");
        let prompt = format!("describe @{}", image_path.display());

        let error = prompt_to_content_blocks(&prompt, Path::new("."))
            .expect_err("unsupported image ref should fail");

        assert!(error.contains("unsupported image format"));
    }

    #[test]
    fn convert_messages_expands_user_text_image_refs() {
        let temp = temp_fixture_dir("convert-message-image-ref");
        let image_path = temp.join("sample.gif");
        std::fs::write(&image_path, [71_u8, 73, 70]).expect("fixture write");
        let messages = vec![ConversationMessage::user_text(format!(
            "inspect @{}",
            image_path.display()
        ))];

        let converted = super::convert_messages(&messages, ApiService::NanoGpt)
            .expect("messages should convert");

        assert_eq!(converted.len(), 1);
        assert!(matches!(
            &converted[0].content[1],
            InputContentBlock::Image { source }
                if source.media_type == "image/gif" && source.data == "R0lG"
        ));
    }

    fn temp_fixture_dir(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should advance")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("pebble-{label}-{unique}"));
        std::fs::create_dir_all(&path).expect("temp dir should exist");
        path
    }

    #[test]
    fn proxy_message_responses_preserve_native_tool_calls() {
        let response = MessageResponse {
            id: "msg_123".to_string(),
            kind: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![
                OutputContentBlock::Text {
                    text: "I will inspect this.\n\n".to_string(),
                },
                OutputContentBlock::ToolUse {
                    id: "toolu_1".to_string(),
                    name: "read_file".to_string(),
                    input: json!({"path":"README.md"}),
                },
            ],
            model: "zai-org/glm-5.1".to_string(),
            stop_reason: Some("end_turn".to_string()),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 10,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                output_tokens: 5,
            },
            request_id: None,
        };

        let events = proxy_response_to_events(response, &mut Vec::new(), &tool_specs())
            .expect("proxy response should convert");
        assert!(matches!(
            &events[0],
            AssistantEvent::TextDelta(text) if text == "I will inspect this.\n\n"
        ));
        assert!(matches!(
            &events[1],
            AssistantEvent::ToolUse { id, name, input }
                if id == "toolu_1"
                    && name == "read_file"
                    && input == "{\"path\":\"README.md\"}"
        ));
    }

    #[test]
    fn retries_proxy_when_reply_only_narrates_tool_intent() {
        let events = vec![
            AssistantEvent::TextDelta(
                "Let me explore the project structure to understand what this is about."
                    .to_string(),
            ),
            AssistantEvent::MessageStop,
        ];

        assert!(should_retry_proxy_tool_prompt(&events));
    }

    #[test]
    fn does_not_retry_proxy_when_tool_call_is_already_present() {
        let events = vec![
            AssistantEvent::TextDelta("Let me inspect that.".to_string()),
            AssistantEvent::ToolUse {
                id: "toolu_1".to_string(),
                name: "read_file".to_string(),
                input: "{\"path\":\"README.md\"}".to_string(),
            },
            AssistantEvent::MessageStop,
        ];

        assert!(!should_retry_proxy_tool_prompt(&events));
    }

    #[test]
    fn read_file_tui_preview_is_compact() {
        let content = (1..=80)
            .map(|line| format!("line {line}"))
            .collect::<Vec<_>>()
            .join("\n");
        let output = serde_json::json!({
            "type": "text",
            "file": {
                "filePath": "/tmp/demo.rs",
                "content": content,
                "numLines": 80,
                "startLine": 1,
                "totalLines": 80
            }
        })
        .to_string();

        let block = render_tool_result_block("read_file", &output);
        let plain = strip_ansi_for_test(&block);

        // The compact block mentions the path and a range, but never spills
        // the full file contents into the transcript.
        assert!(plain.contains("/tmp/demo.rs"));
        assert!(plain.contains("range"));
        assert!(plain.contains("lines 1"));
        assert!(!plain.contains("line 1\n"));
        assert!(!plain.contains("line 80"));
    }

    #[test]
    fn proxy_write_file_xml_is_not_rendered_to_tui() {
        let text = "Now let me create the file: <tool_call name=\"write_file\"><arg name=\"path\">test.md</arg><arg name=\"content\">hello world</arg></tool_call>";
        let mut rendered = Vec::new();
        let mut events = Vec::new();

        append_proxy_text_events(text, &mut rendered, &mut events, &tool_specs())
            .expect("proxy text should parse");

        let rendered_text = String::from_utf8(rendered).expect("rendered bytes should be utf8");
        assert!(rendered_text.trim().is_empty());
        assert!(events.iter().any(|event| matches!(
            event,
            AssistantEvent::ToolUse { name, .. } if name == "write_file"
        )));
    }

    #[test]
    fn bash_tui_preview_truncates_large_stdout() {
        let stdout = (1..=100)
            .map(|line| format!("output {line}"))
            .collect::<Vec<_>>()
            .join("\n");
        let output = serde_json::json!({
            "stdout": stdout,
            "stderr": ""
        })
        .to_string();

        let block = render_tool_result_block("bash", &output);
        let plain = strip_ansi_for_test(&block);

        // The compact block labels the stream and keeps at most a small
        // preview; the tail of the 100-line output must not leak in.
        assert!(plain.contains("stdout"));
        assert!(plain.contains("output 1"));
        assert!(!plain.contains("output 100"));
    }

    #[test]
    fn streaming_tool_use_defers_empty_object_until_json_deltas_arrive() {
        let mut rendered = Vec::new();
        let mut events = Vec::new();
        let mut pending_tool = None;

        push_output_block(
            OutputContentBlock::ToolUse {
                id: "toolu_1".to_string(),
                name: "read_file".to_string(),
                input: json!({}),
            },
            &mut rendered,
            &mut events,
            &mut pending_tool,
            true,
        )
        .expect("tool block should be accepted");

        assert!(rendered.is_empty());
        assert!(events.is_empty());
        assert_eq!(
            pending_tool,
            Some((
                "toolu_1".to_string(),
                "read_file".to_string(),
                String::new()
            ))
        );
    }

    #[test]
    fn streamed_tool_call_start_renders_after_accumulation() {
        let mut rendered = Vec::new();

        render_streamed_tool_call_start(&mut rendered, "read_file", r#"{"path":"README.md"}"#)
            .expect("rendered tool call should succeed");

        let text = String::from_utf8(rendered).expect("rendered bytes should be utf8");
        // The renderer now emits an ANSI-styled, glyph-prefixed header
        // (see `ui::tool_call_header`). Strip ANSI for deterministic
        // substring checks and assert on the stable human-visible parts.
        let plain = strip_ansi_for_test(&text);
        assert!(plain.contains("read_file"));
        assert!(plain.contains("README.md"));
        assert!(!plain.contains("{}"));
    }

    fn strip_ansi_for_test(input: &str) -> String {
        let mut out = String::new();
        let mut chars = input.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '\u{1b}' {
                if chars.peek() == Some(&'[') {
                    chars.next();
                    for next in chars.by_ref() {
                        if next.is_ascii_alphabetic() {
                            break;
                        }
                    }
                }
            } else {
                out.push(ch);
            }
        }
        out
    }

    #[test]
    fn non_stream_response_preserves_empty_object_tool_input() {
        let response = MessageResponse {
            id: "msg_123".to_string(),
            kind: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![OutputContentBlock::ToolUse {
                id: "toolu_1".to_string(),
                name: "read_file".to_string(),
                input: json!({}),
            }],
            model: "zai-org/glm-5.1".to_string(),
            stop_reason: Some("tool_use".to_string()),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 1,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                output_tokens: 1,
            },
            request_id: None,
        };

        let events = response_to_events(response, &mut Vec::new())
            .expect("response conversion should succeed");

        assert!(matches!(
            &events[0],
            AssistantEvent::ToolUse { name, input, .. }
                if name == "read_file" && input == "{}"
        ));
    }

    #[test]
    fn response_to_events_ignores_redacted_thinking_blocks() {
        let events = response_to_events(
            MessageResponse {
                id: "msg_2".to_string(),
                kind: "message".to_string(),
                role: "assistant".to_string(),
                content: vec![
                    OutputContentBlock::RedactedThinking {
                        data: json!({"reason":"hidden"}),
                    },
                    OutputContentBlock::Text {
                        text: "Final answer".to_string(),
                    },
                ],
                model: "zai-org/glm-5.1".to_string(),
                stop_reason: Some("end_turn".to_string()),
                stop_sequence: None,
                usage: Usage {
                    input_tokens: 1,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 0,
                    output_tokens: 1,
                },
                request_id: None,
            },
            &mut Vec::new(),
        )
        .expect("response conversion should succeed");

        assert!(matches!(
            &events[0],
            AssistantEvent::TextDelta(text) if text == "Final answer"
        ));
        assert!(!events.iter().any(|event| matches!(
            event,
            AssistantEvent::ThinkingDelta(_) | AssistantEvent::ThinkingSignature(_)
        )));
    }

    #[test]
    fn strip_markdown_code_fence_removes_wrapping_block() {
        let stripped = strip_markdown_code_fence("```markdown\n# PEBBLE.md\n\nRules\n```")
            .expect("code fence should be removed");

        assert_eq!(stripped, "# PEBBLE.md\n\nRules");
    }

    #[test]
    fn normalize_generated_pebble_md_adds_heading_when_missing() {
        let normalized = normalize_generated_pebble_md("Repository guidance")
            .expect("markdown should normalize");

        assert_eq!(normalized, "# PEBBLE.md\n\nRepository guidance\n");
    }

    #[test]
    fn normalize_generated_pebble_md_prefers_embedded_pebble_heading() {
        let normalized = normalize_generated_pebble_md(
            "Here is the file you requested:\n\n# PEBBLE.md\n\nProject rules",
        )
        .expect("markdown should normalize");

        assert_eq!(normalized, "# PEBBLE.md\n\nProject rules\n");
    }
}

use std::collections::BTreeSet;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use api::{
    resolve_api_key as resolve_nanogpt_api_key, ApiError, ChatCompletionAssistantMessage,
    ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse, ChatCompletionTool,
    ChatCompletionToolChoice, ContentBlockDelta, ImageSource, InputContentBlock, InputMessage,
    MessageRequest, MessageResponse, NanoGptClient, OutputContentBlock,
    StreamEvent as ApiStreamEvent, ThinkingConfig, ToolChoice, ToolDefinition,
    ToolResultContentBlock,
};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use reqwest::blocking::Client as BlockingClient;
use reqwest::header::{HeaderName, HeaderValue};
use serde::Deserialize;
use serde_json::Value as JsonValue;
use sha2::{Digest, Sha256};

use crate::init::initialize_repo;
use crate::input;
use crate::models::{
    default_model_or, max_output_tokens_for_model_or, open_model_picker, open_provider_picker,
    persist_current_model, persist_provider_for_model, persist_proxy_tool_calls,
    provider_for_model, proxy_tool_calls_enabled, validate_provider_for_model,
};
use crate::proxy::{
    build_proxy_system_prompt, convert_messages_for_proxy, parse_proxy_response, parse_proxy_value,
    ProxyCommand, ProxyMessage, ProxySegment, RuntimeToolSpec,
};
use crate::render::{MarkdownStreamState, Spinner, TerminalRenderer};
use commands::{render_slash_command_help, slash_command_specs, SlashCommand};
use compat_harness::{extract_manifest, UpstreamPaths};
use runtime::{
    format_usd, load_system_prompt, mcp_tool_name, pricing_for_model, resolve_sandbox_status,
    spawn_mcp_stdio_process, ApiClient, ApiRequest, AssistantEvent, CompactionConfig, ConfigLoader,
    ConfigSource, ContentBlock, ConversationMessage, ConversationRuntime, JsonRpcId,
    JsonRpcRequest, JsonRpcResponse, McpClientAuth, McpClientBootstrap, McpClientTransport,
    McpInitializeClientInfo, McpInitializeParams, McpListToolsParams, McpListToolsResult,
    McpToolCallParams, McpToolCallResult, McpTransport, MessageRole, PermissionMode,
    PermissionPolicy, PermissionPromptDecision, PermissionPrompter, PermissionRequest,
    RuntimeError, ScopedMcpServerConfig, Session, SessionMetadata, TokenUsage, ToolError,
    ToolExecutor, UsageTracker,
};
use tools::{execute_tool, mvp_tool_specs};

const DEFAULT_MODEL: &str = "zai-org/glm-5.1";
const DEFAULT_MAX_TOKENS: u32 = 4096;
const DEFAULT_THINKING_BUDGET_TOKENS: u32 = 2_048;
const DEFAULT_DATE: &str = "2026-03-31";
const MAX_TOOL_PREVIEW_CHARS: usize = 4_000;
const MAX_TOOL_PREVIEW_LINES: usize = 48;
const MCP_DISCOVERY_TIMEOUT_SECS: u64 = 30;
const COST_WARNING_FRACTION: f64 = 0.8;
const VERSION: &str = env!("CARGO_PKG_VERSION");
const BUILD_TARGET: Option<&str> = option_env!("NANOCODE_BUILD_TARGET");
const IMAGE_REF_PREFIX: &str = "@";
const SELF_UPDATE_REPOSITORY: &str = "nanogpt-community/nanocode-v2";
const SELF_UPDATE_LATEST_RELEASE_URL: &str =
    "https://api.github.com/repos/nanogpt-community/nanocode-v2/releases/latest";
const SELF_UPDATE_USER_AGENT: &str = "nanocode-self-update";
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
        CliAction::Init => run_init()?,
        CliAction::Doctor => run_doctor()?,
        CliAction::SelfUpdate => run_self_update()?,
        CliAction::ResumeSession {
            session_path,
            commands,
        } => resume_session(&session_path, &commands),
        CliAction::Prompt {
            prompt,
            model,
            allowed_tools,
            permission_mode,
            max_cost_usd,
            thinking,
            output_format,
        } => LiveCli::new(
            model,
            true,
            allowed_tools,
            permission_mode,
            max_cost_usd,
            thinking,
            matches!(output_format, CliOutputFormat::Text),
        )?
        .run_turn_with_output(&prompt, output_format)?,
        CliAction::Repl {
            model,
            allowed_tools,
            permission_mode,
            max_cost_usd,
            thinking,
        } => run_repl(
            model,
            allowed_tools,
            permission_mode,
            max_cost_usd,
            thinking,
        )?,
        CliAction::Login { api_key } => login(api_key)?,
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
    Init,
    Doctor,
    SelfUpdate,
    ResumeSession {
        session_path: PathBuf,
        commands: Vec<String>,
    },
    Prompt {
        prompt: String,
        model: String,
        allowed_tools: Option<AllowedToolSet>,
        permission_mode: PermissionMode,
        max_cost_usd: Option<f64>,
        thinking: bool,
        output_format: CliOutputFormat,
    },
    Login {
        api_key: Option<String>,
    },
    Repl {
        model: String,
        allowed_tools: Option<AllowedToolSet>,
        permission_mode: PermissionMode,
        max_cost_usd: Option<f64>,
        thinking: bool,
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
enum McpCommand {
    Status,
    Tools,
    Reload,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct McpServerStatus {
    server_name: String,
    scope: ConfigSource,
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

fn parse_args(args: &[String]) -> Result<CliAction, String> {
    let mut model = resolve_model_alias(&default_model_or(DEFAULT_MODEL)).to_string();
    let mut permission_mode = default_permission_mode();
    let mut max_cost_usd: Option<f64> = None;
    let mut thinking = false;
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
            "--max-cost" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "missing value for --max-cost".to_string())?;
                max_cost_usd = Some(parse_max_cost_arg(value)?);
                index += 2;
            }
            flag if flag.starts_with("--max-cost=") => {
                max_cost_usd = Some(parse_max_cost_arg(&flag[11..])?);
                index += 1;
            }
            "--thinking" => {
                thinking = true;
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
                    max_cost_usd,
                    thinking,
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
            max_cost_usd,
            thinking,
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
        "model" | "models" => parse_model_args(&rest[1..]),
        "provider" | "providers" => parse_provider_args(&rest[1..]),
        "proxy" => parse_proxy_args(&rest[1..]),
        "mcp" => parse_mcp_args(&rest[1..]),
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
                max_cost_usd,
                thinking,
                output_format,
            })
        }
        other if !other.starts_with('/') => Ok(CliAction::Prompt {
            prompt: rest.join(" "),
            model,
            allowed_tools,
            permission_mode,
            max_cost_usd,
            thinking,
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

fn normalize_allowed_tools(values: &[String]) -> Result<Option<AllowedToolSet>, String> {
    if values.is_empty() {
        return Ok(None);
    }

    let canonical_names = base_runtime_tool_specs()
        .into_iter()
        .map(|spec| spec.name)
        .collect::<Vec<_>>();
    let mut allowed = AllowedToolSet::new();

    for value in values {
        for token in value
            .split(|ch: char| ch == ',' || ch.is_whitespace())
            .filter(|token| !token.is_empty())
        {
            let normalized = token.trim().replace('-', "_").to_ascii_lowercase();
            let canonical = match normalized.as_str() {
                "read" => "read_file",
                "write" => "write_file",
                "edit" => "edit_file",
                "glob" => "glob_search",
                "grep" => "grep_search",
                other => other,
            };
            if !canonical_names.iter().any(|name| name == canonical) {
                return Err(format!(
                    "unsupported tool in --allowedTools: {token} (expected one of: {})",
                    canonical_names.join(", ")
                ));
            }
            allowed.insert(canonical.to_string());
        }
    }

    Ok(Some(allowed))
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
    let mut api_key = None;
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--api-key" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "missing value for --api-key".to_string())?;
                api_key = Some(value.clone());
                index += 2;
            }
            flag if flag.starts_with("--api-key=") => {
                api_key = Some(flag[10..].to_string());
                index += 1;
            }
            value if api_key.is_none() => {
                api_key = Some(value.to_string());
                index += 1;
            }
            other => return Err(format!("unexpected login argument: {other}")),
        }
    }

    Ok(CliAction::Login { api_key })
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
    if args.len() > 1 {
        return Err("mcp accepts at most one optional argument".to_string());
    }
    let action = match args.first().map(String::as_str) {
        None | Some("status") => McpCommand::Status,
        Some("tools") => McpCommand::Tools,
        Some("reload") => McpCommand::Reload,
        Some(other) => {
            return Err(format!(
                "mcp accepts one optional argument: status, tools, or reload (got {other})"
            ));
        }
    };
    Ok(CliAction::Mcp { action })
}

fn parse_resume_args(args: &[String]) -> Result<CliAction, String> {
    let session_path = args
        .first()
        .ok_or_else(|| "missing session id or path for --resume".to_string())
        .map(PathBuf::from)?;
    let commands = args[1..].to_vec();
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
    for phase in runtime::BootstrapPlan::nanocode_default().phases() {
        println!("- {phase:?}");
    }
}

fn print_system_prompt(cwd: PathBuf, date: String) {
    match load_system_prompt(cwd, date, env::consts::OS, "unknown") {
        Ok(sections) => println!("{}", sections.join("\n\n")),
        Err(error) => {
            eprintln!("failed to build system prompt: {error}");
            std::process::exit(1);
        }
    }
}

fn resume_session(session_path: &Path, commands: &[String]) {
    let handle = match resolve_session_reference(&session_path.display().to_string()) {
        Ok(handle) => handle,
        Err(error) => {
            eprintln!("failed to resolve session: {error}");
            std::process::exit(1);
        }
    };
    let session = match Session::load_from_path(&handle.path) {
        Ok(session) => session,
        Err(error) => {
            eprintln!("failed to restore session: {error}");
            std::process::exit(1);
        }
    };

    if commands.is_empty() {
        println!(
            "Restored session from {} ({} messages).",
            handle.path.display(),
            session.messages.len()
        );
        return;
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
}

fn login(api_key: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    let api_key = resolve_api_key(api_key)?;
    let credentials_path = save_credentials(&api_key)?;
    println!(
        "Saved NanoGPT credentials to {}",
        credentials_path.display()
    );
    Ok(())
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
    let catalog = load_mcp_catalog(&cwd)?;
    match action {
        McpCommand::Status | McpCommand::Reload => {
            if matches!(action, McpCommand::Reload) {
                println!("Reloaded MCP config from {}", cwd.display());
            }
            print_mcp_status(&catalog);
        }
        McpCommand::Tools => print_mcp_tools(&catalog),
    }
    Ok(())
}

fn resolve_api_key(api_key: Option<String>) -> Result<String, Box<dyn std::error::Error>> {
    match api_key {
        Some(api_key) if !api_key.trim().is_empty() => Ok(api_key),
        Some(_) => Err("NanoGPT API key cannot be empty".into()),
        None => {
            let api_key = read_secret("NanoGPT API key: ")?;
            if api_key.trim().is_empty() {
                return Err("NanoGPT API key cannot be empty".into());
            }
            Ok(api_key)
        }
    }
}

fn save_credentials(api_key: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let config_home = nanocode_config_home()?;
    fs::create_dir_all(&config_home)?;
    let credentials_path = config_home.join("credentials.json");
    fs::write(
        &credentials_path,
        serde_json::to_string_pretty(&serde_json::json!({
            "nanogpt_api_key": api_key
        }))?,
    )?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&credentials_path, fs::Permissions::from_mode(0o600))?;
    }
    Ok(credentials_path)
}

fn nanocode_config_home() -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(path) = env::var_os("NANOCODE_CONFIG_HOME") {
        return Ok(PathBuf::from(path));
    }
    match env::var_os("HOME") {
        Some(home) => Ok(PathBuf::from(home).join(".nanocode")),
        None => Err("could not resolve NANOCODE_CONFIG_HOME or HOME".into()),
    }
}

fn parse_auth_command(input: &str) -> Option<Option<String>> {
    let mut parts = input.split_whitespace();
    let command = parts.next()?;
    if command != "/login" && command != "/auth" {
        return None;
    }

    let remainder = parts.collect::<Vec<_>>().join(" ");
    if remainder.is_empty() {
        Some(None)
    } else {
        Some(Some(remainder))
    }
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

fn parse_thinking_command(input: &str) -> Option<Result<Option<bool>, String>> {
    let mut parts = input.split_whitespace();
    let command = parts.next()?;
    if command != "/thinking" {
        return None;
    }

    let remainder = parts.collect::<Vec<_>>().join(" ");
    let trimmed = remainder.trim();
    Some(match trimmed {
        "" => Ok(None),
        "on" => Ok(Some(true)),
        "off" => Ok(Some(false)),
        other => Err(format!(
            "/thinking accepts one optional argument: on or off (got {other})"
        )),
    })
}

fn parse_mcp_command(input: &str) -> Option<Result<McpCommand, String>> {
    let mut parts = input.split_whitespace();
    let command = parts.next()?;
    if command != "/mcp" {
        return None;
    }

    let remainder = parts.collect::<Vec<_>>().join(" ");
    Some(match remainder.trim() {
        "" | "status" => Ok(McpCommand::Status),
        "tools" => Ok(McpCommand::Tools),
        "reload" => Ok(McpCommand::Reload),
        other => Err(format!(
            "/mcp accepts one optional argument: status, tools, or reload (got {other})"
        )),
    })
}

fn parse_permissions_command(input: &str) -> Option<Result<Option<PermissionMode>, String>> {
    let mut parts = input.split_whitespace();
    let command = parts.next()?;
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

fn parse_max_cost_arg(value: &str) -> Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| format!("invalid value for --max-cost: {value}"))?;
    if !parsed.is_finite() || parsed <= 0.0 {
        return Err(format!(
            "--max-cost must be a positive finite USD amount: {value}"
        ));
    }
    Ok(parsed)
}

fn default_permission_mode() -> PermissionMode {
    env::var("NANOCODE_PERMISSION_MODE")
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
    let result = read_secret_raw(&mut stdout);
    disable_raw_mode()?;
    writeln!(stdout)?;
    result
}

fn read_secret_raw(out: &mut impl Write) -> io::Result<String> {
    let mut secret = String::new();
    loop {
        match event::read()? {
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

fn base_runtime_tool_specs() -> Vec<RuntimeToolSpec> {
    mvp_tool_specs()
        .into_iter()
        .map(RuntimeToolSpec::from)
        .collect()
}

fn available_runtime_tool_specs(mcp_catalog: &McpCatalog) -> Vec<RuntimeToolSpec> {
    let mut specs = base_runtime_tool_specs();
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
            transport: scoped.transport(),
            loaded: false,
            tool_count: 0,
            note: String::new(),
        };

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
                    "{:?} transport is configured but not executable in NanoCode yet",
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

    if !config.mcp().servers().contains_key("nanogpt") {
        if let Some(server) = built_in_nanogpt_mcp_server()? {
            servers.push(("nanogpt".to_string(), server));
        }
    }

    servers.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(servers)
}

fn built_in_nanogpt_mcp_server() -> Result<Option<ScopedMcpServerConfig>, Box<dyn std::error::Error>>
{
    let api_key = match resolve_nanogpt_api_key() {
        Ok(api_key) => api_key,
        Err(ApiError::MissingApiKey) => return Ok(None),
        Err(error) => return Err(error.into()),
    };

    let (command, args) = ensure_nanogpt_mcp_launcher()?;

    let mut env = std::collections::BTreeMap::from([("NANOGPT_API_KEY".to_string(), api_key)]);
    env.entry("NANOGPT_LOG_LEVEL".to_string())
        .or_insert_with(|| "error".to_string());
    for key in [
        "NANOGPT_TIMEOUT_MS",
        "NANOGPT_DEFAULT_MODEL",
        "NANOGPT_BASE_URL",
        "NANOGPT_AUTH_MODE",
        "NANOGPT_MAX_RETRIES",
    ] {
        if let Ok(value) = env::var(key) {
            if !value.is_empty() {
                env.insert(key.to_string(), value);
            }
        }
    }

    let config = runtime::McpServerConfig::Stdio(runtime::McpStdioServerConfig {
        command,
        args,
        env,
        stderr: runtime::McpStdioStderrMode::Null,
    });

    Ok(Some(ScopedMcpServerConfig {
        scope: ConfigSource::User,
        config,
    }))
}

fn ensure_nanogpt_mcp_launcher() -> Result<(String, Vec<String>), Box<dyn std::error::Error>> {
    if let Some(path) = find_path_executable("nanogpt-mcp") {
        return Ok(command_for_binary(&path));
    }

    let install_root = managed_nanogpt_mcp_root()?;
    let binary = managed_nanogpt_mcp_binary_path(&install_root);
    if !binary.exists() {
        install_managed_nanogpt_mcp(&install_root)?;
    }
    if !binary.exists() {
        return Err(format!(
            "NanoGPT MCP launcher was not installed at {}",
            binary.display()
        )
        .into());
    }
    Ok(command_for_binary(&binary))
}

fn managed_nanogpt_mcp_root() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let preferred = nanocode_config_home()?.join("mcp").join("nanogpt");
    if fs::create_dir_all(&preferred).is_ok() {
        return Ok(preferred);
    }

    let fallback = std::env::temp_dir().join("nanocode-mcp").join("nanogpt");
    fs::create_dir_all(&fallback)?;
    Ok(fallback)
}

fn managed_nanogpt_mcp_binary_path(root: &Path) -> PathBuf {
    let binary_name = if cfg!(windows) {
        "nanogpt-mcp.cmd"
    } else {
        "nanogpt-mcp"
    };
    root.join("node_modules").join(".bin").join(binary_name)
}

fn install_managed_nanogpt_mcp(root: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let npm_cache =
        std::env::temp_dir().join(format!("nanocode-npm-cache-install-{}", std::process::id()));
    fs::create_dir_all(&npm_cache)?;

    let npm = if cfg!(windows) { "npm.cmd" } else { "npm" };
    let output = Command::new(npm)
        .arg("install")
        .arg("--prefix")
        .arg(root)
        .arg("@nanogpt/mcp")
        .env("npm_config_cache", &npm_cache)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()?;

    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    let detail = stderr.trim();
    if detail.is_empty() {
        Err(format!("failed to install @nanogpt/mcp into {}", root.display()).into())
    } else {
        Err(format!("failed to install @nanogpt/mcp: {detail}").into())
    }
}

fn find_path_executable(name: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

fn command_for_binary(binary: &Path) -> (String, Vec<String>) {
    if cfg!(windows) {
        (
            "cmd".to_string(),
            vec!["/c".to_string(), binary.to_string_lossy().into_owned()],
        )
    } else {
        (binary.to_string_lossy().into_owned(), Vec::new())
    }
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
                                name: "nanocode".to_string(),
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
                "clientInfo": {"name": "nanocode", "version": VERSION}
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
        other => Err(format!(
            "MCP transport {:?} is not executable in NanoCode yet",
            other
        )
        .into()),
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
                        name: "nanocode".to_string(),
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
                "clientInfo": {"name": "nanocode", "version": VERSION}
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
    let mut request = client.post(url).header("content-type", "application/json");
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
    let mut request = client.post(url).header("content-type", "application/json");
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
        println!("MCP: no servers configured.");
        println!("Add `mcpServers` to `.nanocode/settings.json` to expose MCP tools.");
        return;
    }

    println!(
        "MCP: {} configured server(s), {} exposed MCP tool(s).",
        catalog.servers.len(),
        catalog.tools.len()
    );
    for server in &catalog.servers {
        println!(
            " - {} [{} {:?}] {} tool(s): {}",
            server.server_name,
            config_source_label(server.scope),
            server.transport,
            server.tool_count,
            server.note
        );
    }
}

fn print_mcp_tools(catalog: &McpCatalog) {
    if catalog.tools.is_empty() {
        print_mcp_status(catalog);
        return;
    }

    println!("MCP tools:");
    for tool in &catalog.tools {
        println!(
            " - {} -> {}::{}",
            tool.exposed_name, tool.server_name, tool.upstream_name
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

fn format_status_report(
    model: &str,
    usage: StatusUsage,
    permission_mode: &str,
    provider: &str,
    proxy_tool_calls: bool,
    thinking_enabled: bool,
    max_cost_usd: Option<f64>,
    mcp_catalog: &McpCatalog,
    context: &StatusContext,
) -> String {
    [
        format!(
            "Status\n  Model            {model}\n  Provider         {provider}\n  Permission mode  {permission_mode}\n  Proxy tools      {}\n  Thinking         {}\n  Messages         {}\n  Turns            {}\n  Estimated tokens {}",
            if proxy_tool_calls { "enabled" } else { "disabled" },
            if thinking_enabled { "enabled" } else { "disabled" },
            usage.message_count,
            usage.turns,
            usage.estimated_tokens,
        ),
        format!(
            "Usage\n  Latest total     {}\n  Cumulative input {}\n  Cumulative output {}\n  Cumulative total {}\n  Estimated cost   {}\n  Budget           {}",
            usage.latest.total_tokens(),
            usage.cumulative.input_tokens,
            usage.cumulative.output_tokens,
            usage.cumulative.total_tokens(),
            format_usd(usage_cost_total(model, usage.cumulative)),
            format_budget_line(usage_cost_total(model, usage.cumulative), max_cost_usd),
        ),
        format!(
            "Workspace\n  Cwd              {}\n  Project root     {}\n  Git branch       {}\n  Session          {}\n  Config files     loaded {}/{}\n  Instruction files {}\n  Memory files     {}\n  MCP              servers={} tools={}",
            context.cwd.display(),
            context
                .project_root
                .as_ref()
                .map_or_else(|| "unknown".to_string(), |path| path.display().to_string()),
            context.git_branch.as_deref().unwrap_or("unknown"),
            context.session_path.as_ref().map_or_else(
                || "live-repl".to_string(),
                |path| path.display().to_string()
            ),
            context.loaded_config_files,
            context.discovered_config_files,
            context.instruction_file_count,
            context.memory_file_count,
            mcp_catalog.servers.len(),
            mcp_catalog.tools.len(),
        ),
        format!("Sandbox\n  {}", context.sandbox_summary),
    ]
    .join("\n\n")
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
    let path = cwd.join(".nanocode").join("sessions");
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

fn derive_session_metadata(session: &Session, model: &str) -> SessionMetadata {
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
        compacted.metadata = Some(derive_session_metadata(&compacted, &model));
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
            other => {
                lines.push(format!(
                    "  Unsupported config section '{other}'. Use env, hooks, or model."
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
        lines.push("  No `.nanocode/memory` files discovered.".to_string());
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
    let mut lines = vec![render_slash_command_help()];
    lines.push("Additional NanoCode commands".to_string());
    lines.push("  /login               Save a NanoGPT API key".to_string());
    lines.push("  /auth                Alias for /login".to_string());
    lines.push("  /provider [provider] Set provider override for current model".to_string());
    lines.push("  /proxy [on|off|status] Toggle XML proxy tool calling".to_string());
    lines.push("  /exit                Quit the REPL".to_string());
    lines.push("  /quit                Quit the REPL".to_string());
    lines.join("\n")
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
    max_cost_usd: Option<f64>,
    thinking_enabled: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut cli = LiveCli::new(
        model,
        true,
        allowed_tools,
        permission_mode,
        max_cost_usd,
        thinking_enabled,
        true,
    )?;
    let completions = slash_command_specs()
        .iter()
        .map(|spec| format!("/{}", spec.name))
        .chain(["/login", "/auth", "/provider", "/proxy", "/exit", "/quit"].map(str::to_string))
        .collect();
    let mut editor = input::LineEditor::new("> ", completions);
    println!("NanoCode interactive mode");
    println!("Type /help for commands. Shift+Enter or Ctrl+J inserts a newline.");

    loop {
        let input = match editor.read_line()? {
            input::ReadOutcome::Submit(input) => input,
            input::ReadOutcome::Cancel => continue,
            input::ReadOutcome::Exit => break,
        };
        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }
        editor.push_history(trimmed.to_string());
        if let Some(api_key) = parse_auth_command(trimmed) {
            login(api_key)?;
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
            handle_proxy_runtime_command(&mut cli, mode?)?;
            continue;
        }
        if let Some(enabled) = parse_thinking_command(trimmed) {
            cli.set_thinking(enabled?)?;
            continue;
        }
        if let Some(command) = parse_mcp_command(trimmed) {
            handle_mcp_runtime_command(&mut cli, command?)?;
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
                    SlashCommand::Help => println!("{}", render_repl_help()),
                    SlashCommand::Status => cli.print_status(),
                    SlashCommand::Compact => cli.compact()?,
                    SlashCommand::Thinking { enabled } => cli.set_thinking(enabled)?,
                    SlashCommand::Permissions { mode } => cli.set_permissions(
                        mode.as_deref().map(parse_permission_mode_arg).transpose()?,
                    )?,
                    SlashCommand::Cost => cli.print_cost(),
                    SlashCommand::Clear { confirm } => cli.clear_session(confirm)?,
                    SlashCommand::Resume { session_path } => cli.resume_session(session_path)?,
                    SlashCommand::Config { section } => {
                        println!("{}", render_config_report(section.as_deref())?)
                    }
                    SlashCommand::Memory => println!("{}", render_memory_report()?),
                    SlashCommand::Init => run_init()?,
                    SlashCommand::Diff => println!("{}", render_diff_report()?),
                    SlashCommand::Version => print_version(),
                    SlashCommand::Export { path } => cli.export_session(path.as_deref())?,
                    SlashCommand::Session { action, target } => {
                        cli.handle_session_command(action.as_deref(), target.as_deref())?
                    }
                    SlashCommand::Sessions => println!("{}", render_session_list(&cli.session.id)?),
                    SlashCommand::Unknown(name) => eprintln!("unknown slash command: /{name}"),
                    SlashCommand::Model { .. } | SlashCommand::Mcp { .. } => {
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
    model: String,
    allowed_tools: Option<AllowedToolSet>,
    permission_mode: PermissionMode,
    max_cost_usd: Option<f64>,
    thinking_enabled: bool,
    system_prompt: Vec<String>,
    proxy_tool_calls: bool,
    mcp_catalog: McpCatalog,
    runtime: ConversationRuntime<NanoCodeRuntimeClient, CliToolExecutor>,
    session: SessionHandle,
    render_model_output: bool,
}

impl LiveCli {
    fn new(
        model: String,
        enable_tools: bool,
        allowed_tools: Option<AllowedToolSet>,
        permission_mode: PermissionMode,
        max_cost_usd: Option<f64>,
        thinking_enabled: bool,
        render_model_output: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let system_prompt = build_system_prompt()?;
        let proxy_tool_calls = proxy_tool_calls_enabled();
        let mcp_catalog = load_mcp_catalog(&env::current_dir()?)?;
        let session = create_managed_session_handle()?;
        auto_compact_inactive_sessions(&session.id)?;
        let runtime = build_runtime(
            Session::new(),
            model.clone(),
            system_prompt.clone(),
            enable_tools,
            proxy_tool_calls,
            mcp_catalog.clone(),
            allowed_tools.clone(),
            permission_mode,
            thinking_enabled,
            render_model_output,
        )?;
        let cli = Self {
            model,
            allowed_tools,
            permission_mode,
            max_cost_usd,
            thinking_enabled,
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
        self.enforce_budget_before_turn()?;
        let mut spinner = Spinner::new();
        let mut stdout = io::stdout();
        spinner.tick(
            "Waiting for NanoCode",
            TerminalRenderer::new().color_theme(),
            &mut stdout,
        )?;
        let mut permission_prompter = CliPermissionPrompter::new(self.permission_mode);
        let result = self.runtime.run_turn(input, Some(&mut permission_prompter));
        match result {
            Ok(summary) => {
                spinner.finish(
                    "NanoCode response complete",
                    TerminalRenderer::new().color_theme(),
                    &mut stdout,
                )?;
                self.persist_session()?;
                self.print_budget_notice(summary.usage);
                if let Some(event) = summary.auto_compaction {
                    println!(
                        "{}",
                        format_auto_compaction_notice(event.removed_message_count)
                    );
                }
                println!();
                Ok(())
            }
            Err(error) => {
                spinner.fail(
                    "NanoCode request failed",
                    TerminalRenderer::new().color_theme(),
                    &mut stdout,
                )?;
                Err(Box::new(error))
            }
        }
    }

    fn run_prompt_json(&mut self, input: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.enforce_budget_before_turn()?;
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

    fn print_status(&self) {
        let provider =
            provider_for_model(&self.model).unwrap_or_else(|| "<platform default>".to_string());
        let cumulative = self.runtime.usage().cumulative_usage();
        let latest = self.runtime.usage().current_turn_usage();
        let context = status_context(Some(&self.session.path)).expect("status context should load");
        println!(
            "{}",
            format_status_report(
                &self.model,
                StatusUsage {
                    message_count: self.runtime.session().messages.len(),
                    turns: self.runtime.usage().turns(),
                    latest,
                    cumulative,
                    estimated_tokens: self.runtime.estimated_tokens(),
                },
                self.permission_mode.as_str(),
                &provider,
                self.proxy_tool_calls,
                self.thinking_enabled,
                self.max_cost_usd,
                &self.mcp_catalog,
                &context,
            )
        );
        if self.model == DEFAULT_MODEL {
            println!("status: active model is the default fallback model.");
        } else {
            println!("status: default fallback model={DEFAULT_MODEL}");
        }
    }

    fn compact(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let result = self.runtime.compact(CompactionConfig::default());
        let removed = result.removed_message_count;
        self.runtime = build_runtime(
            result.compacted_session,
            self.model.clone(),
            self.system_prompt.clone(),
            true,
            self.proxy_tool_calls,
            self.mcp_catalog.clone(),
            self.allowed_tools.clone(),
            self.permission_mode,
            self.thinking_enabled,
            self.render_model_output,
        )?;
        self.persist_session()?;
        println!("Compacted {removed} messages.");
        Ok(())
    }

    fn set_model(&mut self, model: String) -> Result<(), Box<dyn std::error::Error>> {
        let model = resolve_model_alias(&model).to_string();
        let session = self.runtime.session().clone();
        persist_current_model(model.clone())?;
        self.runtime = build_runtime(
            session,
            model.clone(),
            self.system_prompt.clone(),
            true,
            self.proxy_tool_calls,
            self.mcp_catalog.clone(),
            self.allowed_tools.clone(),
            self.permission_mode,
            self.thinking_enabled,
            self.render_model_output,
        )?;
        self.model = model.clone();
        self.persist_session()?;
        let provider =
            provider_for_model(&self.model).unwrap_or_else(|| "<platform default>".to_string());
        println!("Switched to model: {model}");
        println!("Provider override for current model: {provider}");
        Ok(())
    }

    fn set_provider(&mut self, provider: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
        let session = self.runtime.session().clone();
        if let Some(provider) = provider.as_deref() {
            validate_provider_for_model(&self.model, provider)?;
        }
        let provider_label = provider
            .clone()
            .unwrap_or_else(|| "<platform default>".to_string());
        persist_provider_for_model(&self.model, provider)?;
        self.runtime = build_runtime(
            session,
            self.model.clone(),
            self.system_prompt.clone(),
            true,
            self.proxy_tool_calls,
            self.mcp_catalog.clone(),
            self.allowed_tools.clone(),
            self.permission_mode,
            self.thinking_enabled,
            self.render_model_output,
        )?;
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
        self.runtime = build_runtime(
            session,
            self.model.clone(),
            self.system_prompt.clone(),
            true,
            enabled,
            self.mcp_catalog.clone(),
            self.allowed_tools.clone(),
            self.permission_mode,
            self.thinking_enabled,
            self.render_model_output,
        )?;
        self.proxy_tool_calls = enabled;
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
        let catalog = load_mcp_catalog(&env::current_dir()?)?;
        self.runtime = build_runtime(
            session,
            self.model.clone(),
            self.system_prompt.clone(),
            true,
            self.proxy_tool_calls,
            catalog.clone(),
            self.allowed_tools.clone(),
            self.permission_mode,
            self.thinking_enabled,
            self.render_model_output,
        )?;
        self.mcp_catalog = catalog;
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

        let session = self.runtime.session().clone();
        self.permission_mode = mode;
        self.runtime = build_runtime(
            session,
            self.model.clone(),
            self.system_prompt.clone(),
            true,
            self.proxy_tool_calls,
            self.mcp_catalog.clone(),
            self.allowed_tools.clone(),
            self.permission_mode,
            self.thinking_enabled,
            self.render_model_output,
        )?;
        self.persist_session()?;
        println!("Permission mode: {}", self.permission_mode.as_str());
        Ok(())
    }

    fn set_thinking(&mut self, enabled: Option<bool>) -> Result<(), Box<dyn std::error::Error>> {
        let Some(enabled) = enabled else {
            println!("{}", format_thinking_report(self.thinking_enabled));
            return Ok(());
        };
        if enabled == self.thinking_enabled {
            println!("{}", format_thinking_report(self.thinking_enabled));
            return Ok(());
        }

        let session = self.runtime.session().clone();
        self.thinking_enabled = enabled;
        self.runtime = build_runtime(
            session,
            self.model.clone(),
            self.system_prompt.clone(),
            true,
            self.proxy_tool_calls,
            self.mcp_catalog.clone(),
            self.allowed_tools.clone(),
            self.permission_mode,
            self.thinking_enabled,
            self.render_model_output,
        )?;
        self.persist_session()?;
        println!("{}", format_thinking_switch_report(self.thinking_enabled));
        Ok(())
    }

    fn print_cost(&self) {
        println!(
            "{}",
            format_cost_report(
                &self.model,
                self.runtime.usage().cumulative_usage(),
                self.max_cost_usd
            )
        );
    }

    fn enforce_budget_before_turn(&self) -> Result<(), Box<dyn std::error::Error>> {
        let Some(limit) = self.max_cost_usd else {
            return Ok(());
        };
        let cumulative = usage_cost_total(&self.model, self.runtime.usage().cumulative_usage());
        if cumulative >= limit {
            return Err(format!(
                "cost budget exceeded before starting turn: cumulative={} budget={}",
                format_usd(cumulative),
                format_usd(limit)
            )
            .into());
        }
        Ok(())
    }

    fn print_budget_notice(&self, usage: TokenUsage) {
        if let Some(message) = budget_notice_message(&self.model, usage, self.max_cost_usd) {
            eprintln!("warning: {message}");
        }
    }

    fn clear_session(&mut self, confirm: bool) -> Result<(), Box<dyn std::error::Error>> {
        if !confirm {
            println!(
                "clear: confirmation required; run /clear --confirm to start a fresh session."
            );
            return Ok(());
        }

        self.session = create_managed_session_handle()?;
        self.runtime = build_runtime(
            Session::new(),
            self.model.clone(),
            self.system_prompt.clone(),
            true,
            self.proxy_tool_calls,
            self.mcp_catalog.clone(),
            self.allowed_tools.clone(),
            self.permission_mode,
            self.thinking_enabled,
            self.render_model_output,
        )?;
        self.persist_session()?;
        println!(
            "Session cleared\n  Mode             fresh session\n  Preserved model  {}\n  Permission mode  {}\n  Session          {}",
            self.model,
            self.permission_mode.as_str(),
            self.session.id,
        );
        Ok(())
    }

    fn resume_session(
        &mut self,
        session_path: Option<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(session_ref) = session_path else {
            println!("Usage: /resume <session-id-or-path>");
            return Ok(());
        };
        let handle = resolve_session_reference(&session_ref)?;
        let session = Session::load_from_path(&handle.path)?;
        let message_count = session.messages.len();
        if let Some(model) = session
            .metadata
            .as_ref()
            .map(|metadata| metadata.model.clone())
        {
            self.model = model;
        }
        self.runtime = build_runtime(
            session,
            self.model.clone(),
            self.system_prompt.clone(),
            true,
            self.proxy_tool_calls,
            self.mcp_catalog.clone(),
            self.allowed_tools.clone(),
            self.permission_mode,
            self.thinking_enabled,
            self.render_model_output,
        )?;
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
                if let Some(model) = session
                    .metadata
                    .as_ref()
                    .map(|metadata| metadata.model.clone())
                {
                    self.model = model;
                }
                self.runtime = build_runtime(
                    session,
                    self.model.clone(),
                    self.system_prompt.clone(),
                    true,
                    self.proxy_tool_calls,
                    self.mcp_catalog.clone(),
                    self.allowed_tools.clone(),
                    self.permission_mode,
                    self.thinking_enabled,
                    self.render_model_output,
                )?;
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

    fn persist_session(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut session = self.runtime.session().clone();
        session.metadata = Some(derive_session_metadata(&session, &self.model));
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

fn build_system_prompt() -> Result<Vec<String>, Box<dyn std::error::Error>> {
    Ok(load_system_prompt(
        env::current_dir()?,
        DEFAULT_DATE,
        env::consts::OS,
        "unknown",
    )?)
}

fn build_runtime_feature_config(
) -> Result<runtime::RuntimeFeatureConfig, Box<dyn std::error::Error>> {
    let cwd = env::current_dir()?;
    Ok(ConfigLoader::default_for(cwd)
        .load()?
        .feature_config()
        .clone())
}

fn build_runtime(
    session: Session,
    model: String,
    system_prompt: Vec<String>,
    enable_tools: bool,
    proxy_tool_calls: bool,
    mcp_catalog: McpCatalog,
    allowed_tools: Option<AllowedToolSet>,
    permission_mode: PermissionMode,
    thinking_enabled: bool,
    render_model_output: bool,
) -> Result<ConversationRuntime<NanoCodeRuntimeClient, CliToolExecutor>, Box<dyn std::error::Error>>
{
    let tool_specs = if enable_tools {
        filter_runtime_tool_specs(
            available_runtime_tool_specs(&mcp_catalog),
            allowed_tools.as_ref(),
        )
    } else {
        Vec::new()
    };
    let mut runtime_prompt = system_prompt;
    if enable_tools && proxy_tool_calls {
        runtime_prompt.push(build_proxy_system_prompt(&tool_specs));
    }
    let permission_policy =
        permission_policy(permission_mode, &available_runtime_tool_specs(&mcp_catalog));
    Ok(ConversationRuntime::new_with_features(
        session,
        NanoCodeRuntimeClient::new(
            model.clone(),
            provider_for_model(&model),
            enable_tools,
            proxy_tool_calls,
            tool_specs.clone(),
            thinking_enabled,
            render_model_output,
        )?,
        CliToolExecutor::new(mcp_catalog, tool_specs, allowed_tools, render_model_output),
        permission_policy,
        runtime_prompt,
        build_runtime_feature_config()?,
    ))
}

struct NanoCodeRuntimeClient {
    runtime: tokio::runtime::Runtime,
    client: NanoGptClient,
    model: String,
    max_output_tokens: u32,
    enable_tools: bool,
    proxy_tool_calls: bool,
    tool_specs: Vec<RuntimeToolSpec>,
    thinking_enabled: bool,
    render_output: bool,
}

impl NanoCodeRuntimeClient {
    fn new(
        model: String,
        provider: Option<String>,
        enable_tools: bool,
        proxy_tool_calls: bool,
        tool_specs: Vec<RuntimeToolSpec>,
        thinking_enabled: bool,
        render_output: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            runtime: tokio::runtime::Runtime::new()?,
            client: NanoGptClient::from_env()?.with_provider(provider.clone()),
            max_output_tokens: max_output_tokens_for_model_or(&model, DEFAULT_MAX_TOKENS),
            model,
            enable_tools,
            proxy_tool_calls,
            tool_specs,
            thinking_enabled,
            render_output,
        })
    }
}

impl ApiClient for NanoCodeRuntimeClient {
    fn stream(&mut self, request: ApiRequest) -> Result<Vec<AssistantEvent>, RuntimeError> {
        if self.proxy_tool_calls {
            return self.stream_via_proxy(request);
        }
        if self.client_provider_active() {
            return self.stream_via_chat_completions(request);
        }

        let message_request = MessageRequest {
            model: self.model.clone(),
            max_tokens: self.max_output_tokens,
            messages: convert_messages(&request.messages)?,
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
            thinking: self
                .thinking_enabled
                .then_some(ThinkingConfig::enabled(DEFAULT_THINKING_BUDGET_TOKENS)),
            stream: true,
        };

        self.runtime.block_on(async {
            let mut stream = self
                .client
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
                            "[nanocode] streaming failed with invalid_response_error{}; retrying non-streaming",
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

            let response = self
                .client
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

impl NanoCodeRuntimeClient {
    fn client_provider_active(&self) -> bool {
        provider_for_model(&self.model).is_some()
    }

    fn stream_via_proxy(
        &mut self,
        request: ApiRequest,
    ) -> Result<Vec<AssistantEvent>, RuntimeError> {
        if self.client_provider_active() {
            return self.stream_proxy_via_chat_completions(request);
        }

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
            thinking: self
                .thinking_enabled
                .then_some(ThinkingConfig::enabled(DEFAULT_THINKING_BUDGET_TOKENS)),
            stream: false,
        };

        self.runtime.block_on(async {
            let response = self
                .client
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
                let retry_response = self
                    .client
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

    fn stream_via_chat_completions(
        &mut self,
        request: ApiRequest,
    ) -> Result<Vec<AssistantEvent>, RuntimeError> {
        let completion_request = ChatCompletionRequest {
            model: self.model.clone(),
            messages: convert_messages_to_chat_completions(
                &request.messages,
                &request.system_prompt,
            ),
            max_tokens: Some(self.max_output_tokens),
            tools: self.enable_tools.then(|| {
                self.tool_specs
                    .iter()
                    .map(|spec| ChatCompletionTool {
                        kind: "function".to_string(),
                        function: api::ChatCompletionFunction {
                            name: spec.name.clone(),
                            description: Some(spec.description.clone()),
                            parameters: Some(spec.input_schema.clone()),
                        },
                    })
                    .collect()
            }),
            tool_choice: self
                .enable_tools
                .then_some(ChatCompletionToolChoice::Mode("auto".to_string())),
            billing_mode: provider_for_model(&self.model).map(|_| "paygo".to_string()),
            stream: false,
        };

        self.runtime.block_on(async {
            let response = self
                .client
                .send_chat_completion(&completion_request)
                .await
                .map_err(|error| RuntimeError::new(error.to_string()))?;
            let mut output: Box<dyn Write> = if self.render_output {
                Box::new(io::stdout())
            } else {
                Box::new(io::sink())
            };
            chat_completion_response_to_events(response, output.as_mut())
        })
    }

    fn stream_proxy_via_chat_completions(
        &mut self,
        request: ApiRequest,
    ) -> Result<Vec<AssistantEvent>, RuntimeError> {
        let mut messages = convert_proxy_messages_to_chat_completions(
            &request.system_prompt,
            convert_messages_for_proxy(&request.messages).map_err(RuntimeError::new)?,
        );
        let base_completion_request = ChatCompletionRequest {
            model: self.model.clone(),
            messages: messages.clone(),
            max_tokens: Some(self.max_output_tokens),
            tools: None,
            tool_choice: None,
            billing_mode: provider_for_model(&self.model).map(|_| "paygo".to_string()),
            stream: false,
        };

        self.runtime.block_on(async {
            let response = self
                .client
                .send_chat_completion(&base_completion_request)
                .await
                .map_err(|error| RuntimeError::new(error.to_string()))?;
            let mut first_render = Vec::new();
            let first_events = proxy_chat_completion_response_to_events(
                response,
                &mut first_render,
                &self.tool_specs,
            )?;

            if should_retry_proxy_tool_prompt(&first_events) {
                messages.push(ChatCompletionMessage {
                    role: "user".to_string(),
                    content: Some(proxy_retry_reminder().to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                });
                let retry_request = ChatCompletionRequest {
                    messages,
                    ..base_completion_request
                };
                let retry_response = self
                    .client
                    .send_chat_completion(&retry_request)
                    .await
                    .map_err(|error| RuntimeError::new(error.to_string()))?;
                let mut output: Box<dyn Write> = if self.render_output {
                    Box::new(io::stdout())
                } else {
                    Box::new(io::sink())
                };
                return proxy_chat_completion_response_to_events(
                    retry_response,
                    output.as_mut(),
                    &self.tool_specs,
                );
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
    writeln!(out, "\n{}", format_tool_call_start(name, input))
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

    if detail.is_empty() {
        format!("→ {name}")
    } else {
        format!("→ {name} {detail}")
    }
}

fn render_thinking_block_summary(
    text: &str,
    out: &mut (impl Write + ?Sized),
) -> Result<(), RuntimeError> {
    writeln!(out, "\n▶ Thinking ({} chars hidden)", text.chars().count())
        .and_then(|_| out.flush())
        .map_err(|error| RuntimeError::new(error.to_string()))
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

fn chat_completion_response_to_events(
    response: ChatCompletionResponse,
    out: &mut (impl Write + ?Sized),
) -> Result<Vec<AssistantEvent>, RuntimeError> {
    let mut events = Vec::new();
    let Some(choice) = response.choices.into_iter().next() else {
        return Err(RuntimeError::new(
            "nanogpt chat completion returned no choices".to_string(),
        ));
    };

    let ChatCompletionAssistantMessage {
        content,
        tool_calls,
        ..
    } = choice.message;

    if let Some(content) = content.filter(|content| !content.is_empty()) {
        write!(out, "{content}")
            .and_then(|_| out.flush())
            .map_err(|error| RuntimeError::new(error.to_string()))?;
        events.push(AssistantEvent::TextDelta(content));
    }

    if let Some(tool_calls) = tool_calls {
        for tool_call in tool_calls {
            events.push(AssistantEvent::ToolUse {
                id: tool_call.id,
                name: tool_call.function.name,
                input: tool_call.function.arguments,
            });
        }
    }

    if let Some(usage) = response.usage {
        events.push(AssistantEvent::Usage(TokenUsage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        }));
    }
    events.push(AssistantEvent::MessageStop);
    Ok(events)
}

fn proxy_chat_completion_response_to_events(
    response: ChatCompletionResponse,
    out: &mut (impl Write + ?Sized),
    tool_specs: &[RuntimeToolSpec],
) -> Result<Vec<AssistantEvent>, RuntimeError> {
    let mut events = Vec::new();
    let Some(choice) = response.choices.into_iter().next() else {
        return Err(RuntimeError::new(
            "nanogpt chat completion returned no choices".to_string(),
        ));
    };
    let ChatCompletionAssistantMessage {
        content,
        tool_calls,
        ..
    } = choice.message;
    append_proxy_text_events(
        content.as_deref().unwrap_or_default(),
        out,
        &mut events,
        tool_specs,
    )?;
    if let Some(tool_calls) = tool_calls {
        for tool_call in tool_calls {
            events.push(AssistantEvent::ToolUse {
                id: tool_call.id,
                name: tool_call.function.name,
                input: tool_call.function.arguments,
            });
        }
    }
    if let Some(usage) = response.usage {
        events.push(AssistantEvent::Usage(TokenUsage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        }));
    }
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
            "creating",
            "writing",
            "saving",
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
    renderer: TerminalRenderer,
    mcp_catalog: McpCatalog,
    tool_specs: Vec<RuntimeToolSpec>,
    allowed_tools: Option<AllowedToolSet>,
    emit_output: bool,
}

impl CliToolExecutor {
    fn new(
        mcp_catalog: McpCatalog,
        tool_specs: Vec<RuntimeToolSpec>,
        allowed_tools: Option<AllowedToolSet>,
        emit_output: bool,
    ) -> Self {
        Self {
            renderer: TerminalRenderer::new(),
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
        let output = if let Some(tool) = self.mcp_catalog.find_tool(tool_name) {
            call_mcp_tool(tool, &value).map_err(|error| ToolError::new(error.to_string()))
        } else {
            execute_tool(tool_name, &value).map_err(ToolError::new)
        }?;
        if self.emit_output {
            let markdown = render_tool_result_markdown(tool_name, &output);
            self.renderer
                .stream_markdown(&markdown, &mut io::stdout())
                .map_err(|error| ToolError::new(error.to_string()))?;
        }
        Ok(output)
    }
}

fn render_tool_result_markdown(tool_name: &str, output: &str) -> String {
    if let Some(markdown) = render_structured_tool_preview(tool_name, output) {
        return markdown;
    }

    let preview = truncate_tool_text(output, MAX_TOOL_PREVIEW_LINES, MAX_TOOL_PREVIEW_CHARS);
    if preview == output {
        format!("### Tool `{tool_name}`\n\n```json\n{output}\n```\n")
    } else {
        format!(
            "### Tool `{tool_name}`\n\n_Output truncated in TUI; full result kept in conversation context._\n\n```json\n{preview}\n```\n"
        )
    }
}

fn render_structured_tool_preview(tool_name: &str, output: &str) -> Option<String> {
    let value = serde_json::from_str::<serde_json::Value>(output).ok()?;
    match tool_name {
        "read_file" => render_read_file_preview(&value),
        "glob_search" => render_glob_search_preview(&value),
        "grep_search" => render_grep_search_preview(&value),
        "bash" => render_bash_preview(&value),
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
    Some(format!(
        "### Tool `read_file`\n\nRead lines {}-{} of {}.\nPath: `{}`\n_File contents hidden in the TUI; full content kept in conversation context._\n",
        start_line.unwrap_or(1),
        end_line.unwrap_or(start_line.unwrap_or(1)),
        total_lines.unwrap_or(num_lines.unwrap_or(0)),
        path
    ))
}

fn render_glob_search_preview(value: &serde_json::Value) -> Option<String> {
    let num_files = value.get("numFiles").and_then(serde_json::Value::as_u64)?;
    let filenames = value.get("filenames")?.as_array()?;
    let preview = filenames
        .iter()
        .filter_map(serde_json::Value::as_str)
        .take(20)
        .collect::<Vec<_>>()
        .join("\n");
    let truncated = filenames.len() > 20
        || value.get("truncated").and_then(serde_json::Value::as_bool) == Some(true);
    Some(format!(
        "### Tool `glob_search`\n\nMatched {} files.\n{}\n```text\n{}\n```\n",
        num_files,
        if truncated {
            "_Preview truncated in TUI; full result kept in conversation context._\n"
        } else {
            ""
        },
        preview
    ))
}

fn render_grep_search_preview(value: &serde_json::Value) -> Option<String> {
    let num_files = value.get("numFiles").and_then(serde_json::Value::as_u64)?;
    let num_matches = value.get("numMatches").and_then(serde_json::Value::as_u64);
    let content = value
        .get("content")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let preview = truncate_tool_text(content, MAX_TOOL_PREVIEW_LINES, MAX_TOOL_PREVIEW_CHARS);
    Some(format!(
        "### Tool `grep_search`\n\nMatched {} files{}.\n{}\n```text\n{}\n```\n",
        num_files,
        num_matches
            .map(|count| format!(", {} matches", count))
            .unwrap_or_default(),
        if preview == content {
            ""
        } else {
            "_Preview truncated in TUI; full result kept in conversation context._\n"
        },
        preview
    ))
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
    let mut sections = Vec::new();
    if !stdout.trim().is_empty() {
        let preview = truncate_tool_text(stdout, MAX_TOOL_PREVIEW_LINES, MAX_TOOL_PREVIEW_CHARS);
        sections.push(format!("**stdout**\n```text\n{}\n```", preview));
        if preview != stdout {
            sections.push(
                "_stdout truncated in TUI; full result kept in conversation context._".to_string(),
            );
        }
    }
    if !stderr.trim().is_empty() {
        let preview = truncate_tool_text(
            stderr,
            MAX_TOOL_PREVIEW_LINES / 2,
            MAX_TOOL_PREVIEW_CHARS / 2,
        );
        sections.push(format!("**stderr**\n```text\n{}\n```", preview));
    }
    if sections.is_empty() {
        return None;
    }
    Some(format!("### Tool `bash`\n\n{}\n", sections.join("\n\n")))
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

fn convert_messages(messages: &[ConversationMessage]) -> Result<Vec<InputMessage>, RuntimeError> {
    let cwd = env::current_dir().map_err(|error| {
        RuntimeError::new(format!("failed to resolve current directory: {error}"))
    })?;
    messages
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
            match content {
                Ok(content) if !content.is_empty() => Some(Ok(InputMessage {
                    role: role.to_string(),
                    content,
                })),
                Ok(_) => None,
                Err(error) => Some(Err(error)),
            }
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
        })
        .collect()
}

fn convert_messages_to_chat_completions(
    messages: &[ConversationMessage],
    system_prompt: &[String],
) -> Vec<ChatCompletionMessage> {
    let mut converted = Vec::new();
    if !system_prompt.is_empty() {
        converted.push(ChatCompletionMessage {
            role: "system".to_string(),
            content: Some(system_prompt.join("\n\n")),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    for message in messages {
        match message.role {
            MessageRole::System => {
                let content = collect_text_blocks(&message.blocks);
                if !content.is_empty() {
                    converted.push(ChatCompletionMessage {
                        role: "system".to_string(),
                        content: Some(content),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
            }
            MessageRole::User => {
                let content = collect_text_blocks(&message.blocks);
                if !content.is_empty() {
                    converted.push(ChatCompletionMessage {
                        role: "user".to_string(),
                        content: Some(content),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
            }
            MessageRole::Assistant => {
                let content = collect_text_blocks(&message.blocks);
                let tool_calls = message
                    .blocks
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::ToolUse { id, name, input } => {
                            Some(api::ChatCompletionToolCall {
                                id: id.clone(),
                                kind: "function".to_string(),
                                function: api::ChatCompletionFunctionCall {
                                    name: name.clone(),
                                    arguments: input.clone(),
                                },
                            })
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                if !content.is_empty() || !tool_calls.is_empty() {
                    converted.push(ChatCompletionMessage {
                        role: "assistant".to_string(),
                        content: (!content.is_empty()).then_some(content),
                        tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
                        tool_call_id: None,
                    });
                }
            }
            MessageRole::Tool => {
                for block in &message.blocks {
                    if let ContentBlock::ToolResult {
                        tool_use_id,
                        output,
                        ..
                    } = block
                    {
                        converted.push(ChatCompletionMessage {
                            role: "tool".to_string(),
                            content: Some(output.clone()),
                            tool_calls: None,
                            tool_call_id: Some(tool_use_id.clone()),
                        });
                    }
                }
            }
        }
    }

    converted
}

fn convert_proxy_messages_to_chat_completions(
    system_prompt: &[String],
    messages: Vec<ProxyMessage>,
) -> Vec<ChatCompletionMessage> {
    let mut converted = Vec::new();
    if !system_prompt.is_empty() {
        converted.push(ChatCompletionMessage {
            role: "system".to_string(),
            content: Some(system_prompt.join("\n\n")),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    converted.extend(messages.into_iter().map(|message| ChatCompletionMessage {
        role: message.role,
        content: Some(message.content),
        tool_calls: None,
        tool_call_id: None,
    }));
    converted
}

fn collect_text_blocks(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            ContentBlock::Thinking { .. } => None,
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn format_thinking_report(enabled: bool) -> String {
    let state = if enabled { "on" } else { "off" };
    let budget = if enabled {
        DEFAULT_THINKING_BUDGET_TOKENS.to_string()
    } else {
        "disabled".to_string()
    };
    format!(
        "Thinking\n  Active mode      {state}\n  Budget tokens    {budget}\n\nUsage\n  Inspect current mode with /thinking\n  Toggle with /thinking on or /thinking off"
    )
}

fn format_thinking_switch_report(enabled: bool) -> String {
    let state = if enabled { "enabled" } else { "disabled" };
    format!(
        "Thinking updated\n  Result           {state}\n  Budget tokens    {}\n  Applies to       subsequent requests",
        if enabled {
            DEFAULT_THINKING_BUDGET_TOKENS.to_string()
        } else {
            "disabled".to_string()
        }
    )
}

fn format_cost_report(model: &str, usage: TokenUsage, max_cost_usd: Option<f64>) -> String {
    let estimate = usage_cost_estimate(model, usage);
    format!(
        "Cost\n  Model            {model}\n  Input tokens     {}\n  Output tokens    {}\n  Cache create     {}\n  Cache read       {}\n  Total tokens     {}\n  Input cost       {}\n  Output cost      {}\n  Cache create usd {}\n  Cache read usd   {}\n  Estimated cost   {}\n  Budget           {}",
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_creation_input_tokens,
        usage.cache_read_input_tokens,
        usage.total_tokens(),
        format_usd(estimate.input_cost_usd),
        format_usd(estimate.output_cost_usd),
        format_usd(estimate.cache_creation_cost_usd),
        format_usd(estimate.cache_read_cost_usd),
        format_usd(estimate.total_cost_usd()),
        format_budget_line(estimate.total_cost_usd(), max_cost_usd),
    )
}

fn usage_cost_estimate(model: &str, usage: TokenUsage) -> runtime::UsageCostEstimate {
    pricing_for_model(model).map_or_else(
        || usage.estimate_cost_usd(),
        |pricing| usage.estimate_cost_usd_with_pricing(pricing),
    )
}

fn usage_cost_total(model: &str, usage: TokenUsage) -> f64 {
    usage_cost_estimate(model, usage).total_cost_usd()
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

fn format_budget_line(cost_usd: f64, max_cost_usd: Option<f64>) -> String {
    match max_cost_usd {
        Some(limit) => format!("{} / {}", format_usd(cost_usd), format_usd(limit)),
        None => format!("{} (unlimited)", format_usd(cost_usd)),
    }
}

fn budget_notice_message(
    model: &str,
    usage: TokenUsage,
    max_cost_usd: Option<f64>,
) -> Option<String> {
    let limit = max_cost_usd?;
    let cost = usage_cost_total(model, usage);
    if cost >= limit {
        Some(format!(
            "cost budget exceeded: cumulative={} budget={}",
            format_usd(cost),
            format_usd(limit)
        ))
    } else if cost >= limit * COST_WARNING_FRACTION {
        Some(format!(
            "approaching cost budget: cumulative={} budget={}",
            format_usd(cost),
            format_usd(limit)
        ))
    } else {
        None
    }
}

fn run_resume_command(
    session_path: &Path,
    session: &Session,
    command: &SlashCommand,
) -> Result<ResumeCommandOutcome, Box<dyn std::error::Error>> {
    match command {
        SlashCommand::Help => Ok(ResumeCommandOutcome {
            session: session.clone(),
            message: Some(render_repl_help()),
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
            Ok(ResumeCommandOutcome {
                session: session.clone(),
                message: Some(format_status_report(
                    session
                        .metadata
                        .as_ref()
                        .map(|metadata| metadata.model.as_str())
                        .unwrap_or("restored-session"),
                    StatusUsage {
                        message_count: session.messages.len(),
                        turns: tracker.turns(),
                        latest: tracker.current_turn_usage(),
                        cumulative: usage,
                        estimated_tokens: 0,
                    },
                    default_permission_mode().as_str(),
                    "<platform default>",
                    false,
                    false,
                    None,
                    &McpCatalog::default(),
                    &status_context(Some(session_path))?,
                )),
            })
        }
        SlashCommand::Cost => {
            let model = session
                .metadata
                .as_ref()
                .map(|metadata| metadata.model.as_str())
                .unwrap_or(DEFAULT_MODEL);
            Ok(ResumeCommandOutcome {
                session: session.clone(),
                message: Some(format_cost_report(
                    model,
                    UsageTracker::from_session(session).cumulative_usage(),
                    None,
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
        SlashCommand::Init => Ok(ResumeCommandOutcome {
            session: session.clone(),
            message: Some(initialize_repo(&env::current_dir()?)?.render()),
        }),
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
        | SlashCommand::Mcp { .. }
        | SlashCommand::Permissions { .. }
        | SlashCommand::Thinking { .. }
        | SlashCommand::Session { .. }
        | SlashCommand::Sessions
        | SlashCommand::Unknown(_) => Err("unsupported resumed slash command".into()),
    }
}

fn print_help() {
    println!("nanocode");
    println!();
    println!("Usage:");
    println!(
        "  nanocode [--model MODEL] [--permission-mode MODE] [--max-cost USD] [--thinking] [--allowedTools TOOL[,TOOL...]]"
    );
    println!("                                               Start interactive REPL");
    println!("  nanocode login [--api-key KEY]              Save a NanoGPT API key");
    println!("  nanocode model [MODEL_ID]                   Choose or persist a default model");
    println!(
        "  nanocode provider [PROVIDER_ID|default]     Choose a provider for the active model"
    );
    println!("  nanocode proxy [on|off|status]              Toggle XML tool-call proxy mode");
    println!(
        "  nanocode mcp [status|tools|reload]          Inspect configured MCP servers and tools"
    );
    println!("  nanocode init                               Create starter NanoCode project files");
    println!("  nanocode doctor                             Run local environment diagnostics");
    println!("  nanocode self-update                        Update from GitHub releases");
    println!("  nanocode --resume SESSION_ID_OR_PATH [/status] [/compact] [...]");
    println!("                                               Resume a saved session and run slash commands");
    println!(
        "  nanocode prompt [--model MODEL] [--permission-mode MODE] [--max-cost USD] [--thinking] [--output-format text|json] TEXT"
    );
    println!(
        "                                               Send one prompt and stream the response"
    );
    println!(
        "  nanocode [--model MODEL] [--permission-mode MODE] [--max-cost USD] [--thinking] [--output-format text|json] TEXT"
    );
    println!(
        "                                               Shorthand non-interactive prompt mode"
    );
    println!("  nanocode dump-manifests");
    println!("  nanocode bootstrap-plan");
    println!("  nanocode system-prompt [--cwd PATH] [--date YYYY-MM-DD]");
    println!("  nanocode --version");
    println!(
        "  --permission-mode MODE                     read-only, workspace-write, or danger-full-access"
    );
    println!(
        "  --max-cost USD                             Warn at 80% of budget and stop at/exceeding the budget"
    );
    println!(
        "  --thinking                                 Enable extended thinking with the default budget"
    );
    println!(
        "  --output-format FORMAT                     Non-interactive output format: text or json"
    );
}

fn print_version() {
    println!("{}", render_version_report());
}

fn run_init() -> Result<(), Box<dyn std::error::Error>> {
    let cwd = env::current_dir()?;
    println!("{}", initialize_repo(&cwd)?.render());
    Ok(())
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
                Some("GitHub latest release endpoint returned no published release for nanogpt-community/nanocode-v2."),
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
            let mut names = vec![format!("nanocode-{target}")];
            if env::consts::OS == "windows" {
                names.push(format!("nanocode-{target}.exe"));
            }
            names
        })
        .collect::<Vec<_>>();
    if env::consts::OS == "windows" {
        candidates.push("nanocode.exe".to_string());
    }
    candidates.push("nanocode".to_string());
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
        append_proxy_text_events, available_runtime_tool_specs, extract_first_json_object,
        filter_runtime_tool_specs, parse_args, parse_auth_command, parse_checksum_for_asset,
        parse_mcp_command, parse_model_command, parse_provider_command, parse_proxy_command,
        parse_tool_input_value, prompt_to_content_blocks, proxy_chat_completion_response_to_events,
        proxy_response_to_events, push_output_block, render_streamed_tool_call_start,
        render_tool_result_markdown, render_update_report, resolve_model_alias, response_to_events,
        should_retry_proxy_tool_prompt, AssistantEvent, CliAction, CliOutputFormat, GitHubRelease,
        GitHubReleaseAsset, McpCatalog, McpCommand, RuntimeToolSpec, DEFAULT_MODEL,
    };
    use crate::proxy::ProxyCommand;
    use api::{
        ChatCompletionAssistantMessage, ChatCompletionChoice, ChatCompletionFunctionCall,
        ChatCompletionResponse, ChatCompletionToolCall, ChatCompletionUsage, InputContentBlock,
        MessageResponse, OutputContentBlock, Usage,
    };
    use runtime::{ContentBlock, ConversationMessage, MessageRole, PermissionMode};
    use serde_json::json;
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, OnceLock};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env test lock should not be poisoned")
    }

    fn with_isolated_config_home<T>(run: impl FnOnce() -> T) -> T {
        let _guard = env_lock();
        let root = std::env::temp_dir().join(format!(
            "nanocode-main-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time should be after epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).expect("config dir should exist");
        std::env::set_var("NANOCODE_CONFIG_HOME", &root);
        let output = run();
        std::env::remove_var("NANOCODE_CONFIG_HOME");
        std::fs::remove_dir_all(root).expect("temp config dir should be removed");
        output
    }

    fn tool_specs() -> Vec<RuntimeToolSpec> {
        available_runtime_tool_specs(&McpCatalog::default())
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
                    max_cost_usd: None,
                    thinking: false,
                }
            );
        });
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
                    max_cost_usd: None,
                    thinking: false,
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
                    max_cost_usd: None,
                    thinking: false,
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
                    max_cost_usd: None,
                    thinking: false,
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
                    max_cost_usd: None,
                    thinking: false,
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
                    max_cost_usd: None,
                    thinking: false,
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
                    max_cost_usd: None,
                    thinking: false,
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
                    api_key: Some("nano-key".to_string()),
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
                    session_path: PathBuf::from("session.json"),
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
                    session_path: PathBuf::from("session.json"),
                    commands: vec!["/status".to_string(), "/export".to_string()],
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
        let manifest = "abc123 *nanocode-aarch64-apple-darwin\ndef456 other-file\n";
        assert_eq!(
            parse_checksum_for_asset(manifest, "nanocode-aarch64-apple-darwin"),
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
                browser_download_url: "https://example.invalid/nanocode".to_string(),
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
        assert!(report.contains("nanogpt-community/nanocode-v2"));
    }

    #[test]
    fn filtered_tool_specs_respect_allowlist() {
        let allowed = ["read_file", "grep_search"]
            .into_iter()
            .map(str::to_string)
            .collect();
        let filtered = filter_runtime_tool_specs(
            available_runtime_tool_specs(&McpCatalog::default()),
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
        assert_eq!(parse_auth_command("/login"), Some(None));
        assert_eq!(
            parse_auth_command("/auth nano-key"),
            Some(Some("nano-key".to_string()))
        );
        assert_eq!(parse_auth_command("/status"), None);
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
    fn resolves_known_nanocode_model_aliases() {
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

        let converted = super::convert_messages(&messages).expect("messages should convert");
        assert_eq!(converted.len(), 3);
        assert_eq!(converted[1].role, "assistant");
        assert_eq!(converted[2].role, "user");
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

        let converted = super::convert_messages(&messages).expect("messages should convert");

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
        let path = std::env::temp_dir().join(format!("nanocode-{label}-{unique}"));
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
    fn proxy_chat_completion_responses_preserve_native_tool_calls() {
        let response = ChatCompletionResponse {
            id: "chatcmpl_123".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatCompletionAssistantMessage {
                    role: "assistant".to_string(),
                    content: Some("I will inspect this.\n\n".to_string()),
                    tool_calls: Some(vec![ChatCompletionToolCall {
                        id: "call_1".to_string(),
                        kind: "function".to_string(),
                        function: ChatCompletionFunctionCall {
                            name: "read_file".to_string(),
                            arguments: "{\"path\":\"README.md\"}".to_string(),
                        },
                    }]),
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            usage: Some(ChatCompletionUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            }),
            service_tier: None,
        };

        let events =
            proxy_chat_completion_response_to_events(response, &mut Vec::new(), &tool_specs())
                .expect("proxy chat completion should convert");
        assert!(matches!(
            &events[0],
            AssistantEvent::TextDelta(text) if text == "I will inspect this.\n\n"
        ));
        assert!(matches!(
            &events[1],
            AssistantEvent::ToolUse { id, name, input }
                if id == "call_1"
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

        let markdown = render_tool_result_markdown("read_file", &output);

        assert!(markdown.contains("### Tool `read_file`"));
        assert!(markdown.contains("Path: `/tmp/demo.rs`"));
        assert!(markdown.contains("File contents hidden in the TUI"));
        assert!(!markdown.contains("line 1"));
        assert!(!markdown.contains("line 80"));
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

        let markdown = render_tool_result_markdown("bash", &output);

        assert!(markdown.contains("### Tool `bash`"));
        assert!(markdown.contains("stdout truncated in TUI"));
        assert!(!markdown.contains("output 100"));
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
        assert!(text.contains("→ read_file README.md"));
        assert!(!text.contains("{}"));
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
}

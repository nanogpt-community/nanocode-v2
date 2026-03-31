mod input;
mod models;
mod proxy;
mod render;

use std::env;
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};

use api::{
    ApiError, ChatCompletionAssistantMessage, ChatCompletionMessage, ChatCompletionRequest,
    ChatCompletionResponse, ChatCompletionTool, ChatCompletionToolChoice, ContentBlockDelta,
    InputContentBlock, InputMessage, MessageRequest, MessageResponse, NanoGptClient,
    OutputContentBlock, StreamEvent as ApiStreamEvent, ToolChoice, ToolDefinition,
    ToolResultContentBlock,
};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};

use commands::handle_slash_command;
use compat_harness::{extract_manifest, UpstreamPaths};
use models::{
    default_model_or, max_output_tokens_for_model_or, open_model_picker, open_provider_picker,
    persist_current_model, persist_provider_for_model, persist_proxy_tool_calls,
    provider_for_model, proxy_tool_calls_enabled, validate_provider_for_model,
};
use proxy::{
    build_proxy_system_prompt, convert_messages_for_proxy, parse_proxy_response, parse_proxy_value,
    ProxyCommand, ProxyMessage, ProxySegment,
};
use render::{Spinner, TerminalRenderer};
use runtime::{
    load_system_prompt, ApiClient, ApiRequest, AssistantEvent, CompactionConfig, ContentBlock,
    ConversationMessage, ConversationRuntime, MessageRole, PermissionMode, PermissionPolicy,
    RuntimeError, Session, TokenUsage, ToolError, ToolExecutor,
};
use tools::{execute_tool, mvp_tool_specs};

const DEFAULT_MODEL: &str = "zai-org/glm-5.1";
const DEFAULT_MAX_TOKENS: u32 = 4096;
const DEFAULT_DATE: &str = "2026-03-31";
const MAX_TOOL_PREVIEW_CHARS: usize = 4_000;
const MAX_TOOL_PREVIEW_LINES: usize = 48;
const VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    match parse_args(&args)? {
        CliAction::DumpManifests => dump_manifests(),
        CliAction::BootstrapPlan => print_bootstrap_plan(),
        CliAction::PrintSystemPrompt { cwd, date } => print_system_prompt(cwd, date),
        CliAction::Model { model } => handle_model_action(model)?,
        CliAction::Provider { provider } => handle_provider_action(provider)?,
        CliAction::Proxy { mode } => handle_proxy_action(mode)?,
        CliAction::ResumeSession {
            session_path,
            command,
        } => resume_session(&session_path, command),
        CliAction::Prompt { prompt, model } => LiveCli::new(model, false)?.run_turn(&prompt)?,
        CliAction::Repl { model } => run_repl(model)?,
        CliAction::Login { api_key } => login(api_key)?,
        CliAction::Help => print_help(),
        CliAction::Version => print_version(),
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
    ResumeSession {
        session_path: PathBuf,
        command: Option<String>,
    },
    Prompt {
        prompt: String,
        model: String,
    },
    Login {
        api_key: Option<String>,
    },
    Repl {
        model: String,
    },
    Help,
    Version,
}

fn parse_args(args: &[String]) -> Result<CliAction, String> {
    let mut model = default_model_or(DEFAULT_MODEL);
    let mut rest = Vec::new();
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--model" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "missing value for --model".to_string())?;
                model = value.clone();
                index += 2;
            }
            flag if flag.starts_with("--model=") => {
                model = flag[8..].to_string();
                index += 1;
            }
            other => {
                rest.push(other.to_string());
                index += 1;
            }
        }
    }

    if rest.is_empty() {
        return Ok(CliAction::Repl { model });
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
        "prompt" => {
            let prompt = rest[1..].join(" ");
            if prompt.trim().is_empty() {
                return Err("prompt subcommand requires a prompt string".to_string());
            }
            Ok(CliAction::Prompt { prompt, model })
        }
        other => Err(format!("unknown subcommand: {other}")),
    }
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

fn parse_resume_args(args: &[String]) -> Result<CliAction, String> {
    let session_path = args
        .first()
        .ok_or_else(|| "missing session path for --resume".to_string())
        .map(PathBuf::from)?;
    let command = args.get(1).cloned();
    if args.len() > 2 {
        return Err("--resume accepts at most one trailing slash command".to_string());
    }
    Ok(CliAction::ResumeSession {
        session_path,
        command,
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

fn resume_session(session_path: &Path, command: Option<String>) {
    let session = match Session::load_from_path(session_path) {
        Ok(session) => session,
        Err(error) => {
            eprintln!("failed to restore session: {error}");
            std::process::exit(1);
        }
    };

    match command {
        Some(command) if command.starts_with('/') => {
            let Some(result) = handle_slash_command(
                &command,
                &session,
                CompactionConfig {
                    max_estimated_tokens: 0,
                    ..CompactionConfig::default()
                },
            ) else {
                eprintln!("unknown slash command: {command}");
                std::process::exit(2);
            };
            if let Err(error) = result.session.save_to_path(session_path) {
                eprintln!("failed to persist resumed session: {error}");
                std::process::exit(1);
            }
            println!("{}", result.message);
        }
        Some(other) => {
            eprintln!("unsupported resumed command: {other}");
            std::process::exit(2);
        }
        None => {
            println!(
                "Restored session from {} ({} messages).",
                session_path.display(),
                session.messages.len()
            );
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

fn run_repl(model: String) -> Result<(), Box<dyn std::error::Error>> {
    let mut cli = LiveCli::new(model, true)?;
    let editor = input::LineEditor::new("› ");
    println!("NanoCode interactive mode");
    println!("Type /help for commands. Shift+Enter or Ctrl+J inserts a newline.");

    while let Some(input) = editor.read_line()? {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }
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
        match trimmed {
            "/exit" | "/quit" => break,
            "/help" => {
                println!("Available commands:");
                println!("  /help    Show help");
                println!("  /login   Save a NanoGPT API key");
                println!("  /auth    Alias for /login");
                println!("  /model   Choose the active NanoGPT model");
                println!("  /provider Choose a provider override for the current model only");
                println!("  /proxy   Toggle XML tool-call proxy mode (or /proxy on|off|status)");
                println!("  /status  Show session status");
                println!("  /compact Compact session history");
                println!("  /exit    Quit the REPL");
            }
            "/status" => cli.print_status(),
            "/compact" => cli.compact()?,
            _ => cli.run_turn(trimmed)?,
        }
    }

    Ok(())
}

struct LiveCli {
    model: String,
    system_prompt: Vec<String>,
    proxy_tool_calls: bool,
    runtime: ConversationRuntime<NanoCodeRuntimeClient, CliToolExecutor>,
}

impl LiveCli {
    fn new(model: String, enable_tools: bool) -> Result<Self, Box<dyn std::error::Error>> {
        let system_prompt = build_system_prompt()?;
        let proxy_tool_calls = proxy_tool_calls_enabled();
        let runtime = build_runtime(
            Session::new(),
            model.clone(),
            system_prompt.clone(),
            enable_tools,
            proxy_tool_calls,
        )?;
        Ok(Self {
            model,
            system_prompt,
            proxy_tool_calls,
            runtime,
        })
    }

    fn run_turn(&mut self, input: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut spinner = Spinner::new();
        let mut stdout = io::stdout();
        spinner.tick(
            "Waiting for NanoCode",
            TerminalRenderer::new().color_theme(),
            &mut stdout,
        )?;
        let result = self.runtime.run_turn(input, None);
        match result {
            Ok(_) => {
                spinner.finish(
                    "NanoCode response complete",
                    TerminalRenderer::new().color_theme(),
                    &mut stdout,
                )?;
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

    fn print_status(&self) {
        let usage = self.runtime.usage().cumulative_usage();
        let provider =
            provider_for_model(&self.model).unwrap_or_else(|| "<platform default>".to_string());
        println!(
            "status: model={} provider_for_current_model={} proxy_tool_calls={} messages={} turns={} input_tokens={} output_tokens={}",
            self.model,
            provider,
            if self.proxy_tool_calls {
                "enabled"
            } else {
                "disabled"
            },
            self.runtime.session().messages.len(),
            self.runtime.usage().turns(),
            usage.input_tokens,
            usage.output_tokens
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
        )?;
        println!("Compacted {removed} messages.");
        Ok(())
    }

    fn set_model(&mut self, model: String) -> Result<(), Box<dyn std::error::Error>> {
        let session = self.runtime.session().clone();
        persist_current_model(model.clone())?;
        self.runtime = build_runtime(
            session,
            model.clone(),
            self.system_prompt.clone(),
            true,
            self.proxy_tool_calls,
        )?;
        self.model = model.clone();
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
        )?;
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
        )?;
        self.proxy_tool_calls = enabled;
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

fn build_system_prompt() -> Result<Vec<String>, Box<dyn std::error::Error>> {
    Ok(load_system_prompt(
        env::current_dir()?,
        DEFAULT_DATE,
        env::consts::OS,
        "unknown",
    )?)
}

fn build_runtime(
    session: Session,
    model: String,
    system_prompt: Vec<String>,
    enable_tools: bool,
    proxy_tool_calls: bool,
) -> Result<ConversationRuntime<NanoCodeRuntimeClient, CliToolExecutor>, Box<dyn std::error::Error>>
{
    let mut runtime_prompt = system_prompt;
    if enable_tools && proxy_tool_calls {
        runtime_prompt.push(build_proxy_system_prompt(&mvp_tool_specs()));
    }
    Ok(ConversationRuntime::new(
        session,
        NanoCodeRuntimeClient::new(
            model.clone(),
            provider_for_model(&model),
            enable_tools,
            proxy_tool_calls,
        )?,
        CliToolExecutor::new(),
        permission_policy_from_env(),
        runtime_prompt,
    ))
}

struct NanoCodeRuntimeClient {
    runtime: tokio::runtime::Runtime,
    client: NanoGptClient,
    model: String,
    max_output_tokens: u32,
    enable_tools: bool,
    proxy_tool_calls: bool,
}

impl NanoCodeRuntimeClient {
    fn new(
        model: String,
        provider: Option<String>,
        enable_tools: bool,
        proxy_tool_calls: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            runtime: tokio::runtime::Runtime::new()?,
            client: NanoGptClient::from_env()?.with_provider(provider.clone()),
            max_output_tokens: max_output_tokens_for_model_or(&model, DEFAULT_MAX_TOKENS),
            model,
            enable_tools,
            proxy_tool_calls,
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
            messages: convert_messages(&request.messages),
            system: (!request.system_prompt.is_empty()).then(|| request.system_prompt.join("\n\n")),
            tools: self.enable_tools.then(|| {
                mvp_tool_specs()
                    .into_iter()
                    .map(|spec| ToolDefinition {
                        name: spec.name.to_string(),
                        description: Some(spec.description.to_string()),
                        input_schema: spec.input_schema,
                    })
                    .collect()
            }),
            tool_choice: self.enable_tools.then_some(ToolChoice::Auto),
            stream: true,
        };

        self.runtime.block_on(async {
            let mut stream = self
                .client
                .stream_message(&message_request)
                .await
                .map_err(|error| RuntimeError::new(error.to_string()))?;
            let mut stdout = io::stdout();
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
                            push_output_block(block, &mut stdout, &mut events, &mut pending_tool)?;
                        }
                    }
                    ApiStreamEvent::ContentBlockStart(start) => {
                        push_output_block(
                            start.content_block,
                            &mut stdout,
                            &mut events,
                            &mut pending_tool,
                        )?;
                    }
                    ApiStreamEvent::ContentBlockDelta(delta) => match delta.delta {
                        ContentBlockDelta::TextDelta { text } => {
                            if !text.is_empty() {
                                write!(stdout, "{text}")
                                    .and_then(|_| stdout.flush())
                                    .map_err(|error| RuntimeError::new(error.to_string()))?;
                                events.push(AssistantEvent::TextDelta(text));
                            }
                        }
                        ContentBlockDelta::InputJsonDelta { partial_json } => {
                            if let Some((_, _, input)) = &mut pending_tool {
                                input.push_str(&partial_json);
                            }
                        }
                    },
                    ApiStreamEvent::ContentBlockStop(_) => {
                        if let Some((id, name, input)) = pending_tool.take() {
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
                        events.push(AssistantEvent::MessageStop);
                    }
                }
            }

            if !stream_fallback_requested
                && !saw_stop
                && events.iter().any(|event| {
                    matches!(event, AssistantEvent::TextDelta(text) if !text.is_empty())
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
            response_to_events(response, &mut stdout)
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
            stream: false,
        };

        self.runtime.block_on(async {
            let response = self
                .client
                .send_message(&base_message_request)
                .await
                .map_err(|error| RuntimeError::new(error.to_string()))?;
            let mut first_render = Vec::new();
            let first_events = proxy_response_to_events(response, &mut first_render)?;

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
                let mut stdout = io::stdout();
                return proxy_response_to_events(retry_response, &mut stdout);
            }

            let mut stdout = io::stdout();
            stdout
                .write_all(&first_render)
                .and_then(|_| stdout.flush())
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
                mvp_tool_specs()
                    .into_iter()
                    .map(|spec| ChatCompletionTool {
                        kind: "function".to_string(),
                        function: api::ChatCompletionFunction {
                            name: spec.name.to_string(),
                            description: Some(spec.description.to_string()),
                            parameters: Some(spec.input_schema),
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
            chat_completion_response_to_events(response, &mut io::stdout())
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
            let first_events = proxy_chat_completion_response_to_events(response, &mut first_render)?;

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
                let mut stdout = io::stdout();
                return proxy_chat_completion_response_to_events(retry_response, &mut stdout);
            }

            let mut stdout = io::stdout();
            stdout
                .write_all(&first_render)
                .and_then(|_| stdout.flush())
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
    let intent_prefix = ["let me ", "i'll ", "i will ", "first, i'll", "first i ", "let's "];
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
        && tool_intent
            .iter()
            .any(|phrase| normalized.contains(phrase))
}

fn push_output_block(
    block: OutputContentBlock,
    out: &mut impl Write,
    events: &mut Vec<AssistantEvent>,
    pending_tool: &mut Option<(String, String, String)>,
) -> Result<(), RuntimeError> {
    match block {
        OutputContentBlock::Text { text } => {
            if !text.is_empty() {
                write!(out, "{text}")
                    .and_then(|_| out.flush())
                    .map_err(|error| RuntimeError::new(error.to_string()))?;
                events.push(AssistantEvent::TextDelta(text));
            }
        }
        OutputContentBlock::ToolUse { id, name, input } => {
            *pending_tool = Some((id, name, input.to_string()));
        }
    }
    Ok(())
}

fn response_to_events(
    response: MessageResponse,
    out: &mut impl Write,
) -> Result<Vec<AssistantEvent>, RuntimeError> {
    let mut events = Vec::new();
    let mut pending_tool = None;

    for block in response.content {
        push_output_block(block, out, &mut events, &mut pending_tool)?;
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
    out: &mut impl Write,
) -> Result<Vec<AssistantEvent>, RuntimeError> {
    let mut events = Vec::new();
    for block in response.content {
        match block {
            OutputContentBlock::Text { text } => {
                append_proxy_text_events(&text, out, &mut events)?;
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
    out: &mut impl Write,
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
    out: &mut impl Write,
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
    append_proxy_text_events(content.as_deref().unwrap_or_default(), out, &mut events)?;
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
    out: &mut impl Write,
    events: &mut Vec<AssistantEvent>,
) -> Result<(), RuntimeError> {
    for segment in parse_proxy_response(text, &mvp_tool_specs()).map_err(RuntimeError::new)? {
        match segment {
            ProxySegment::Text(text) => {
                if !text.is_empty() {
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

struct CliToolExecutor {
    renderer: TerminalRenderer,
}

impl CliToolExecutor {
    fn new() -> Self {
        Self {
            renderer: TerminalRenderer::new(),
        }
    }
}

impl ToolExecutor for CliToolExecutor {
    fn execute(&mut self, tool_name: &str, input: &str) -> Result<String, ToolError> {
        let value = parse_tool_input_value(tool_name, input)?;
        match execute_tool(tool_name, &value) {
            Ok(output) => {
                let markdown = render_tool_result_markdown(tool_name, &output);
                self.renderer
                    .stream_markdown(&markdown, &mut io::stdout())
                    .map_err(|error| ToolError::new(error.to_string()))?;
                Ok(output)
            }
            Err(error) => Err(ToolError::new(error)),
        }
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
    let content = file.get("content").and_then(serde_json::Value::as_str).unwrap_or("");
    let preview = truncate_tool_text(content, MAX_TOOL_PREVIEW_LINES, MAX_TOOL_PREVIEW_CHARS);
    let end_line = match (start_line, num_lines) {
        (Some(start), Some(count)) if count > 0 => Some(start + count - 1),
        _ => start_line,
    };
    Some(format!(
        "### Tool `read_file`\n\n{} lines {}-{} of {}.\nPath: `{}`\n{}\n```text\n{}\n```\n",
        if preview == content { "Showing" } else { "Previewing" },
        start_line.unwrap_or(1),
        end_line.unwrap_or(start_line.unwrap_or(1)),
        total_lines.unwrap_or(num_lines.unwrap_or(0)),
        path,
        if preview == content {
            ""
        } else {
            "_Output truncated in TUI; full file content kept in conversation context._\n"
        },
        preview
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
    let truncated = filenames.len() > 20 || value.get("truncated").and_then(serde_json::Value::as_bool) == Some(true);
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
    let content = value.get("content").and_then(serde_json::Value::as_str).unwrap_or("");
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
    let stdout = value.get("stdout").and_then(serde_json::Value::as_str).unwrap_or("");
    let stderr = value.get("stderr").and_then(serde_json::Value::as_str).unwrap_or("");
    let mut sections = Vec::new();
    if !stdout.trim().is_empty() {
        let preview = truncate_tool_text(stdout, MAX_TOOL_PREVIEW_LINES, MAX_TOOL_PREVIEW_CHARS);
        sections.push(format!(
            "**stdout**\n```text\n{}\n```",
            preview
        ));
        if preview != stdout {
            sections.push("_stdout truncated in TUI; full result kept in conversation context._".to_string());
        }
    }
    if !stderr.trim().is_empty() {
        let preview = truncate_tool_text(stderr, MAX_TOOL_PREVIEW_LINES / 2, MAX_TOOL_PREVIEW_CHARS / 2);
        sections.push(format!(
            "**stderr**\n```text\n{}\n```",
            preview
        ));
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
                output.push_str(&line.chars().take(remaining.saturating_sub(1)).collect::<String>());
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

fn parse_tool_input_value(tool_name: &str, input: &str) -> Result<serde_json::Value, ToolError> {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(input) {
        return Ok(value);
    }

    if let Some(value) = extract_first_json_object(input) {
        return Ok(value);
    }

    if input.contains("<tool_call") {
        let tool_use = parse_proxy_response(input, &mvp_tool_specs())
            .map_err(ToolError::new)?
            .into_iter()
            .find_map(|segment| match segment {
                ProxySegment::ToolUse { name, input, .. } if name == tool_name => Some(input),
                _ => None,
            })
            .ok_or_else(|| ToolError::new("proxy tool call did not contain a matching tool"))?;
        return serde_json::from_str(&tool_use)
            .map_err(|error| ToolError::new(format!("invalid recovered proxy tool JSON: {error}")));
    }

    if input.contains("<arg") {
        let wrapped = format!("<tool_call name=\"{tool_name}\">{input}</tool_call>");
        let tool_use = parse_proxy_response(&wrapped, &mvp_tool_specs())
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

fn permission_policy_from_env() -> PermissionPolicy {
    let mode =
        env::var("NANOCODE_PERMISSION_MODE").unwrap_or_else(|_| "workspace-write".to_string());
    match mode.as_str() {
        "read-only" => PermissionPolicy::new(PermissionMode::Deny)
            .with_tool_mode("read_file", PermissionMode::Allow)
            .with_tool_mode("glob_search", PermissionMode::Allow)
            .with_tool_mode("grep_search", PermissionMode::Allow),
        _ => PermissionPolicy::new(PermissionMode::Allow),
    }
}

fn convert_messages(messages: &[ConversationMessage]) -> Vec<InputMessage> {
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
                .map(|block| match block {
                    ContentBlock::Text { text } => InputContentBlock::Text { text: text.clone() },
                    ContentBlock::ToolUse { id, name, input } => InputContentBlock::ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        input: serde_json::from_str(input)
                            .unwrap_or_else(|_| serde_json::json!({ "raw": input })),
                    },
                    ContentBlock::ToolResult {
                        tool_use_id,
                        output,
                        is_error,
                        ..
                    } => InputContentBlock::ToolResult {
                        tool_use_id: tool_use_id.clone(),
                        content: vec![ToolResultContentBlock::Text {
                            text: output.clone(),
                        }],
                        is_error: *is_error,
                    },
                })
                .collect::<Vec<_>>();
            (!content.is_empty()).then(|| InputMessage {
                role: role.to_string(),
                content,
            })
        })
        .collect()
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
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn print_help() {
    println!("nanocode");
    println!();
    println!("Usage:");
    println!("  nanocode [--model MODEL]                     Start interactive REPL");
    println!("  nanocode login [--api-key KEY]              Save a NanoGPT API key");
    println!("  nanocode model [MODEL_ID]                   Choose or persist a default model");
    println!(
        "  nanocode provider [PROVIDER_ID|default]     Choose a provider for the active model"
    );
    println!("  nanocode proxy [on|off|status]              Toggle XML tool-call proxy mode");
    println!(
        "  nanocode [--model MODEL] prompt TEXT        Send one prompt and stream the response"
    );
    println!("  nanocode dump-manifests");
    println!("  nanocode bootstrap-plan");
    println!("  nanocode system-prompt [--cwd PATH] [--date YYYY-MM-DD]");
    println!("  nanocode --resume SESSION.json [/compact]");
    println!("  nanocode --version");
}

fn print_version() {
    println!("{VERSION}");
}

#[cfg(test)]
mod tests {
    use api::{
        ChatCompletionAssistantMessage, ChatCompletionChoice, ChatCompletionFunctionCall,
        ChatCompletionResponse, ChatCompletionToolCall, ChatCompletionUsage, MessageResponse,
        OutputContentBlock, Usage,
    };
    use super::{
        extract_first_json_object, parse_args, parse_auth_command, parse_model_command,
        parse_provider_command, parse_proxy_command, parse_tool_input_value,
        proxy_chat_completion_response_to_events, proxy_response_to_events,
        render_tool_result_markdown, AssistantEvent, should_retry_proxy_tool_prompt, CliAction,
        DEFAULT_MODEL,
    };
    use crate::proxy::ProxyCommand;
    use runtime::{ContentBlock, ConversationMessage, MessageRole};
    use serde_json::json;
    use std::path::PathBuf;
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

    #[test]
    fn defaults_to_repl_when_no_args() {
        with_isolated_config_home(|| {
            assert_eq!(
                parse_args(&[]).expect("args should parse"),
                CliAction::Repl {
                    model: DEFAULT_MODEL.to_string(),
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
                }
            );
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
                    command: Some("/compact".to_string()),
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

        let converted = super::convert_messages(&messages);
        assert_eq!(converted.len(), 3);
        assert_eq!(converted[1].role, "assistant");
        assert_eq!(converted[2].role, "user");
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

        let events = proxy_response_to_events(response, &mut Vec::new())
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

        let events = proxy_chat_completion_response_to_events(response, &mut Vec::new())
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
        assert!(markdown.contains("full file content kept in conversation context"));
        assert!(!markdown.contains("line 80"));
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
}

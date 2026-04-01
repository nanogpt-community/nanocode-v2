use runtime::{compact_session, CompactionConfig, Session};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandManifestEntry {
    pub name: String,
    pub source: CommandSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandSource {
    Builtin,
    InternalOnly,
    FeatureGated,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CommandRegistry {
    entries: Vec<CommandManifestEntry>,
}

impl CommandRegistry {
    #[must_use]
    pub fn new(entries: Vec<CommandManifestEntry>) -> Self {
        Self { entries }
    }

    #[must_use]
    pub fn entries(&self) -> &[CommandManifestEntry] {
        &self.entries
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlashCommandSpec {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub summary: &'static str,
    pub argument_hint: Option<&'static str>,
    pub resume_supported: bool,
}

const SLASH_COMMAND_SPECS: &[SlashCommandSpec] = &[
    SlashCommandSpec {
        name: "help",
        aliases: &[],
        summary: "Show available slash commands",
        argument_hint: None,
        resume_supported: true,
    },
    SlashCommandSpec {
        name: "status",
        aliases: &[],
        summary: "Show current session status",
        argument_hint: None,
        resume_supported: true,
    },
    SlashCommandSpec {
        name: "compact",
        aliases: &[],
        summary: "Compact local session history",
        argument_hint: None,
        resume_supported: true,
    },
    SlashCommandSpec {
        name: "thinking",
        aliases: &[],
        summary: "Show or toggle extended thinking",
        argument_hint: Some("[on|off]"),
        resume_supported: false,
    },
    SlashCommandSpec {
        name: "model",
        aliases: &[],
        summary: "Show or switch the active model",
        argument_hint: Some("[model]"),
        resume_supported: false,
    },
    SlashCommandSpec {
        name: "mcp",
        aliases: &[],
        summary: "Inspect configured MCP servers or list exposed MCP tools",
        argument_hint: Some("[status|tools|reload]"),
        resume_supported: false,
    },
    SlashCommandSpec {
        name: "permissions",
        aliases: &[],
        summary: "Show or switch the active permission mode",
        argument_hint: Some("[read-only|workspace-write|danger-full-access]"),
        resume_supported: false,
    },
    SlashCommandSpec {
        name: "clear",
        aliases: &[],
        summary: "Start a fresh local session",
        argument_hint: Some("[--confirm]"),
        resume_supported: true,
    },
    SlashCommandSpec {
        name: "cost",
        aliases: &[],
        summary: "Show cumulative token usage for this session",
        argument_hint: None,
        resume_supported: true,
    },
    SlashCommandSpec {
        name: "resume",
        aliases: &[],
        summary: "Load a saved session into the REPL",
        argument_hint: Some("<session-path>"),
        resume_supported: false,
    },
    SlashCommandSpec {
        name: "config",
        aliases: &[],
        summary: "Inspect NanoCode config files or merged sections",
        argument_hint: Some("[env|hooks|model|plugins]"),
        resume_supported: true,
    },
    SlashCommandSpec {
        name: "memory",
        aliases: &[],
        summary: "Inspect loaded NanoCode instruction memory files",
        argument_hint: None,
        resume_supported: true,
    },
    SlashCommandSpec {
        name: "init",
        aliases: &[],
        summary: "Create a starter NANOCODE.md for this repo",
        argument_hint: None,
        resume_supported: true,
    },
    SlashCommandSpec {
        name: "diff",
        aliases: &[],
        summary: "Show git diff for current workspace changes",
        argument_hint: None,
        resume_supported: true,
    },
    SlashCommandSpec {
        name: "version",
        aliases: &[],
        summary: "Show CLI version and build information",
        argument_hint: None,
        resume_supported: true,
    },
    SlashCommandSpec {
        name: "export",
        aliases: &[],
        summary: "Export the current conversation to a file",
        argument_hint: Some("[file]"),
        resume_supported: true,
    },
    SlashCommandSpec {
        name: "session",
        aliases: &[],
        summary: "List or switch managed local sessions",
        argument_hint: Some("[list|switch <session-id>]"),
        resume_supported: false,
    },
    SlashCommandSpec {
        name: "sessions",
        aliases: &[],
        summary: "List recent managed local sessions",
        argument_hint: None,
        resume_supported: false,
    },
    SlashCommandSpec {
        name: "plugins",
        aliases: &["plugin", "marketplace"],
        summary: "Manage NanoCode plugins",
        argument_hint: Some(
            "[list|install <path>|enable <id>|disable <id>|uninstall <id>|update <id>]",
        ),
        resume_supported: false,
    },
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlashCommand {
    Help,
    Status,
    Compact,
    Thinking {
        enabled: Option<bool>,
    },
    Model {
        model: Option<String>,
    },
    Mcp {
        action: Option<String>,
    },
    Permissions {
        mode: Option<String>,
    },
    Clear {
        confirm: bool,
    },
    Cost,
    Resume {
        session_path: Option<String>,
    },
    Config {
        section: Option<String>,
    },
    Memory,
    Init,
    Diff,
    Version,
    Export {
        path: Option<String>,
    },
    Session {
        action: Option<String>,
        target: Option<String>,
    },
    Sessions,
    Plugins {
        action: Option<String>,
        target: Option<String>,
    },
    Unknown(String),
}

impl SlashCommand {
    #[must_use]
    pub fn parse(input: &str) -> Option<Self> {
        let trimmed = input.trim();
        if !trimmed.starts_with('/') {
            return None;
        }

        let mut parts = trimmed.trim_start_matches('/').split_whitespace();
        let command = parts.next().unwrap_or_default();
        Some(match command {
            "help" => Self::Help,
            "status" => Self::Status,
            "compact" => Self::Compact,
            "thinking" => Self::Thinking {
                enabled: match parts.next() {
                    Some("on") => Some(true),
                    Some("off") => Some(false),
                    Some(_) | None => None,
                },
            },
            "model" => Self::Model {
                model: parts.next().map(ToOwned::to_owned),
            },
            "mcp" => Self::Mcp {
                action: parts.next().map(ToOwned::to_owned),
            },
            "permissions" => Self::Permissions {
                mode: parts.next().map(ToOwned::to_owned),
            },
            "clear" => Self::Clear {
                confirm: parts.next() == Some("--confirm"),
            },
            "cost" => Self::Cost,
            "resume" => Self::Resume {
                session_path: parts.next().map(ToOwned::to_owned),
            },
            "config" => Self::Config {
                section: parts.next().map(ToOwned::to_owned),
            },
            "memory" => Self::Memory,
            "init" => Self::Init,
            "diff" => Self::Diff,
            "version" => Self::Version,
            "export" => Self::Export {
                path: parts.next().map(ToOwned::to_owned),
            },
            "session" => Self::Session {
                action: parts.next().map(ToOwned::to_owned),
                target: parts.next().map(ToOwned::to_owned),
            },
            "sessions" => Self::Sessions,
            "plugin" | "plugins" | "marketplace" => Self::Plugins {
                action: parts.next().map(ToOwned::to_owned),
                target: {
                    let remainder = parts.collect::<Vec<_>>().join(" ");
                    (!remainder.is_empty()).then_some(remainder)
                },
            },
            other => Self::Unknown(other.to_string()),
        })
    }
}

#[must_use]
pub fn slash_command_specs() -> &'static [SlashCommandSpec] {
    SLASH_COMMAND_SPECS
}

#[must_use]
pub fn resume_supported_slash_commands() -> Vec<&'static SlashCommandSpec> {
    slash_command_specs()
        .iter()
        .filter(|spec| spec.resume_supported)
        .collect()
}

#[must_use]
pub fn render_slash_command_help() -> String {
    let mut lines = vec![
        "Slash commands".to_string(),
        "  [resume] means the command also works with --resume SESSION.json".to_string(),
    ];
    for spec in slash_command_specs() {
        let name = match spec.argument_hint {
            Some(argument_hint) => format!("/{} {}", spec.name, argument_hint),
            None => format!("/{}", spec.name),
        };
        let aliases = if spec.aliases.is_empty() {
            String::new()
        } else {
            format!(" (aliases: {})", spec.aliases.join(", "))
        };
        let resume = if spec.resume_supported {
            " [resume]"
        } else {
            ""
        };
        lines.push(format!(
            "  {name:<20} {}{}{}",
            spec.summary, aliases, resume
        ));
    }
    lines.join("\n")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlashCommandResult {
    pub message: String,
    pub session: Session,
}

#[must_use]
pub fn handle_slash_command(
    input: &str,
    session: &Session,
    compaction: CompactionConfig,
) -> Option<SlashCommandResult> {
    match SlashCommand::parse(input)? {
        SlashCommand::Compact => {
            let result = compact_session(session, compaction);
            let message = if result.removed_message_count == 0 {
                "Compaction skipped: session is below the compaction threshold.".to_string()
            } else {
                format!(
                    "Compacted {} messages into a resumable system summary.",
                    result.removed_message_count
                )
            };
            Some(SlashCommandResult {
                message,
                session: result.compacted_session,
            })
        }
        SlashCommand::Help => Some(SlashCommandResult {
            message: render_slash_command_help(),
            session: session.clone(),
        }),
        SlashCommand::Status
        | SlashCommand::Thinking { .. }
        | SlashCommand::Model { .. }
        | SlashCommand::Mcp { .. }
        | SlashCommand::Permissions { .. }
        | SlashCommand::Clear { .. }
        | SlashCommand::Cost
        | SlashCommand::Resume { .. }
        | SlashCommand::Config { .. }
        | SlashCommand::Memory
        | SlashCommand::Init
        | SlashCommand::Diff
        | SlashCommand::Version
        | SlashCommand::Export { .. }
        | SlashCommand::Session { .. }
        | SlashCommand::Sessions
        | SlashCommand::Plugins { .. }
        | SlashCommand::Unknown(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        handle_slash_command, render_slash_command_help, resume_supported_slash_commands,
        slash_command_specs, SlashCommand,
    };
    use runtime::{CompactionConfig, ContentBlock, ConversationMessage, MessageRole, Session};

    #[test]
    fn parses_supported_slash_commands() {
        assert_eq!(SlashCommand::parse("/help"), Some(SlashCommand::Help));
        assert_eq!(SlashCommand::parse(" /status "), Some(SlashCommand::Status));
        assert_eq!(
            SlashCommand::parse("/model claude-opus"),
            Some(SlashCommand::Model {
                model: Some("claude-opus".to_string()),
            })
        );
        assert_eq!(
            SlashCommand::parse("/model"),
            Some(SlashCommand::Model { model: None })
        );
        assert_eq!(
            SlashCommand::parse("/permissions read-only"),
            Some(SlashCommand::Permissions {
                mode: Some("read-only".to_string()),
            })
        );
        assert_eq!(
            SlashCommand::parse("/clear"),
            Some(SlashCommand::Clear { confirm: false })
        );
        assert_eq!(
            SlashCommand::parse("/clear --confirm"),
            Some(SlashCommand::Clear { confirm: true })
        );
        assert_eq!(SlashCommand::parse("/cost"), Some(SlashCommand::Cost));
        assert_eq!(
            SlashCommand::parse("/resume session.json"),
            Some(SlashCommand::Resume {
                session_path: Some("session.json".to_string()),
            })
        );
        assert_eq!(
            SlashCommand::parse("/config"),
            Some(SlashCommand::Config { section: None })
        );
        assert_eq!(
            SlashCommand::parse("/config env"),
            Some(SlashCommand::Config {
                section: Some("env".to_string())
            })
        );
        assert_eq!(SlashCommand::parse("/memory"), Some(SlashCommand::Memory));
        assert_eq!(SlashCommand::parse("/init"), Some(SlashCommand::Init));
        assert_eq!(SlashCommand::parse("/diff"), Some(SlashCommand::Diff));
        assert_eq!(SlashCommand::parse("/version"), Some(SlashCommand::Version));
        assert_eq!(
            SlashCommand::parse("/export notes.txt"),
            Some(SlashCommand::Export {
                path: Some("notes.txt".to_string())
            })
        );
        assert_eq!(
            SlashCommand::parse("/session switch abc123"),
            Some(SlashCommand::Session {
                action: Some("switch".to_string()),
                target: Some("abc123".to_string())
            })
        );
        assert_eq!(
            SlashCommand::parse("/plugins enable demo@external"),
            Some(SlashCommand::Plugins {
                action: Some("enable".to_string()),
                target: Some("demo@external".to_string())
            })
        );
        assert_eq!(
            SlashCommand::parse("/marketplace list"),
            Some(SlashCommand::Plugins {
                action: Some("list".to_string()),
                target: None
            })
        );
    }

    #[test]
    fn renders_help_from_shared_specs() {
        let help = render_slash_command_help();
        assert!(help.contains("works with --resume SESSION.json"));
        assert!(help.contains("/help"));
        assert!(help.contains("/status"));
        assert!(help.contains("/compact"));
        assert!(help.contains("/thinking [on|off]"));
        assert!(help.contains("/model [model]"));
        assert!(help.contains("/mcp [status|tools|reload]"));
        assert!(help.contains("/permissions [read-only|workspace-write|danger-full-access]"));
        assert!(help.contains("/clear [--confirm]"));
        assert!(help.contains("/cost"));
        assert!(help.contains("/resume <session-path>"));
        assert!(help.contains("/config [env|hooks|model|plugins]"));
        assert!(help.contains("/memory"));
        assert!(help.contains("/init"));
        assert!(help.contains("/diff"));
        assert!(help.contains("/version"));
        assert!(help.contains("/export [file]"));
        assert!(help.contains("/session [list|switch <session-id>]"));
        assert!(help.contains("/sessions"));
        assert!(help.contains(
            "/plugins [list|install <path>|enable <id>|disable <id>|uninstall <id>|update <id>]"
        ));
        assert_eq!(slash_command_specs().len(), 19);
        assert_eq!(resume_supported_slash_commands().len(), 11);
    }

    #[test]
    fn compacts_sessions_via_slash_command() {
        let root = std::env::temp_dir().join(format!(
            "commands-compact-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).expect("temp dir");
        let previous = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(&root).expect("set cwd");

        let session = Session {
            version: 1,
            messages: vec![
                ConversationMessage::user_text("a ".repeat(200)),
                ConversationMessage::assistant(vec![ContentBlock::Text {
                    text: "b ".repeat(200),
                }]),
                ConversationMessage::tool_result("1", "bash", "ok ".repeat(200), false),
                ConversationMessage::assistant(vec![ContentBlock::Text {
                    text: "recent".to_string(),
                }]),
            ],
            metadata: None,
        };

        let result = handle_slash_command(
            "/compact",
            &session,
            CompactionConfig {
                preserve_recent_messages: 2,
                max_estimated_tokens: 1,
            },
        )
        .expect("slash command should be handled");

        std::env::set_current_dir(previous).expect("restore cwd");
        let _ = std::fs::remove_dir_all(root);

        assert!(result.message.contains("Compacted 2 messages"));
        assert_eq!(result.session.messages[0].role, MessageRole::System);
    }

    #[test]
    fn help_command_is_non_mutating() {
        let session = Session::new();
        let result = handle_slash_command("/help", &session, CompactionConfig::default())
            .expect("help command should be handled");
        assert_eq!(result.session, session);
        assert!(result.message.contains("Slash commands"));
    }

    #[test]
    fn ignores_unknown_or_runtime_bound_slash_commands() {
        let session = Session::new();
        assert!(handle_slash_command("/unknown", &session, CompactionConfig::default()).is_none());
        assert!(handle_slash_command("/status", &session, CompactionConfig::default()).is_none());
        assert!(
            handle_slash_command("/model claude", &session, CompactionConfig::default()).is_none()
        );
        assert!(handle_slash_command(
            "/permissions read-only",
            &session,
            CompactionConfig::default()
        )
        .is_none());
        assert!(handle_slash_command("/clear", &session, CompactionConfig::default()).is_none());
        assert!(
            handle_slash_command("/clear --confirm", &session, CompactionConfig::default())
                .is_none()
        );
        assert!(handle_slash_command("/cost", &session, CompactionConfig::default()).is_none());
        assert!(handle_slash_command(
            "/resume session.json",
            &session,
            CompactionConfig::default()
        )
        .is_none());
        assert!(handle_slash_command("/config", &session, CompactionConfig::default()).is_none());
        assert!(
            handle_slash_command("/config env", &session, CompactionConfig::default()).is_none()
        );
        assert!(handle_slash_command("/diff", &session, CompactionConfig::default()).is_none());
        assert!(handle_slash_command("/version", &session, CompactionConfig::default()).is_none());
        assert!(
            handle_slash_command("/export note.txt", &session, CompactionConfig::default())
                .is_none()
        );
        assert!(
            handle_slash_command("/session list", &session, CompactionConfig::default()).is_none()
        );
        assert!(
            handle_slash_command("/plugins list", &session, CompactionConfig::default()).is_none()
        );
    }
}

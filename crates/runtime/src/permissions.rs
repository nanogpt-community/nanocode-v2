use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PermissionMode {
    ReadOnly,
    WorkspaceWrite,
    DangerFullAccess,
}

impl PermissionMode {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::WorkspaceWrite => "workspace-write",
            Self::DangerFullAccess => "danger-full-access",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionOverride {
    Allow,
    Deny,
    Ask,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PermissionContext {
    override_decision: Option<PermissionOverride>,
    override_reason: Option<String>,
}

impl PermissionContext {
    #[must_use]
    pub fn new(
        override_decision: Option<PermissionOverride>,
        override_reason: Option<String>,
    ) -> Self {
        Self {
            override_decision,
            override_reason,
        }
    }

    #[must_use]
    pub fn override_decision(&self) -> Option<PermissionOverride> {
        self.override_decision
    }

    #[must_use]
    pub fn override_reason(&self) -> Option<&str> {
        self.override_reason.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermissionRequest {
    pub tool_name: String,
    pub input: String,
    pub current_mode: PermissionMode,
    pub required_mode: PermissionMode,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PermissionPromptDecision {
    Allow,
    Deny { reason: String },
}

pub trait PermissionPrompter {
    fn decide(&mut self, request: &PermissionRequest) -> PermissionPromptDecision;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PermissionOutcome {
    Allow,
    Deny { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermissionPolicy {
    active_mode: PermissionMode,
    tool_requirements: BTreeMap<String, PermissionMode>,
}

impl PermissionPolicy {
    #[must_use]
    pub fn new(active_mode: PermissionMode) -> Self {
        Self {
            active_mode,
            tool_requirements: BTreeMap::new(),
        }
    }

    #[must_use]
    pub fn with_tool_requirement(
        mut self,
        tool_name: impl Into<String>,
        required_mode: PermissionMode,
    ) -> Self {
        self.tool_requirements
            .insert(tool_name.into(), required_mode);
        self
    }

    #[must_use]
    pub fn active_mode(&self) -> PermissionMode {
        self.active_mode
    }

    #[must_use]
    pub fn required_mode_for(&self, tool_name: &str) -> PermissionMode {
        self.tool_requirements
            .get(tool_name)
            .copied()
            .unwrap_or(PermissionMode::DangerFullAccess)
    }

    #[must_use]
    pub fn authorize(
        &self,
        tool_name: &str,
        input: &str,
        prompter: Option<&mut dyn PermissionPrompter>,
    ) -> PermissionOutcome {
        self.authorize_with_context(tool_name, input, &PermissionContext::default(), prompter)
    }

    #[must_use]
    pub fn authorize_with_context(
        &self,
        tool_name: &str,
        input: &str,
        context: &PermissionContext,
        prompter: Option<&mut dyn PermissionPrompter>,
    ) -> PermissionOutcome {
        let current_mode = self.active_mode();
        let required_mode = self.required_mode_for(tool_name);
        match context.override_decision() {
            Some(PermissionOverride::Allow) => return PermissionOutcome::Allow,
            Some(PermissionOverride::Deny) => {
                return PermissionOutcome::Deny {
                    reason: context.override_reason().map_or_else(
                        || format!("tool '{tool_name}' denied by hook"),
                        ToOwned::to_owned,
                    ),
                };
            }
            Some(PermissionOverride::Ask) => {
                return Self::prompt_or_deny(
                    tool_name,
                    input,
                    current_mode,
                    required_mode,
                    Some(context.override_reason().map_or_else(
                        || format!("tool '{tool_name}' requires approval due to hook guidance"),
                        ToOwned::to_owned,
                    )),
                    prompter,
                );
            }
            None => {}
        }

        if current_mode >= required_mode {
            return PermissionOutcome::Allow;
        }

        if current_mode == PermissionMode::WorkspaceWrite
            && required_mode == PermissionMode::DangerFullAccess
        {
            return Self::prompt_or_deny(
                tool_name,
                input,
                current_mode,
                required_mode,
                Some(format!(
                    "tool '{tool_name}' requires approval to escalate from {} to {}",
                    current_mode.as_str(),
                    required_mode.as_str()
                )),
                prompter,
            );
        }

        PermissionOutcome::Deny {
            reason: format!(
                "tool '{tool_name}' requires {} permission; current mode is {}",
                required_mode.as_str(),
                current_mode.as_str()
            ),
        }
    }

    fn prompt_or_deny(
        tool_name: &str,
        input: &str,
        current_mode: PermissionMode,
        required_mode: PermissionMode,
        reason: Option<String>,
        mut prompter: Option<&mut dyn PermissionPrompter>,
    ) -> PermissionOutcome {
        let request = PermissionRequest {
            tool_name: tool_name.to_string(),
            input: input.to_string(),
            current_mode,
            required_mode,
            reason: reason.clone(),
        };

        match prompter.as_mut() {
            Some(prompter) => match prompter.decide(&request) {
                PermissionPromptDecision::Allow => PermissionOutcome::Allow,
                PermissionPromptDecision::Deny { reason } => PermissionOutcome::Deny { reason },
            },
            None => PermissionOutcome::Deny {
                reason: reason.unwrap_or_else(|| {
                    format!(
                        "tool '{tool_name}' requires approval to run while mode is {}",
                        current_mode.as_str()
                    )
                }),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PermissionContext, PermissionMode, PermissionOutcome, PermissionOverride,
        PermissionPolicy, PermissionPromptDecision, PermissionPrompter, PermissionRequest,
    };

    struct RecordingPrompter {
        seen: Vec<PermissionRequest>,
        allow: bool,
    }

    impl PermissionPrompter for RecordingPrompter {
        fn decide(&mut self, request: &PermissionRequest) -> PermissionPromptDecision {
            self.seen.push(request.clone());
            if self.allow {
                PermissionPromptDecision::Allow
            } else {
                PermissionPromptDecision::Deny {
                    reason: "not now".to_string(),
                }
            }
        }
    }

    #[test]
    fn allows_tools_when_active_mode_meets_requirement() {
        let policy = PermissionPolicy::new(PermissionMode::WorkspaceWrite)
            .with_tool_requirement("read_file", PermissionMode::ReadOnly)
            .with_tool_requirement("write_file", PermissionMode::WorkspaceWrite);

        assert_eq!(
            policy.authorize("read_file", "{}", None),
            PermissionOutcome::Allow
        );
        assert_eq!(
            policy.authorize("write_file", "{}", None),
            PermissionOutcome::Allow
        );
    }

    #[test]
    fn denies_read_only_escalations_without_prompt() {
        let policy = PermissionPolicy::new(PermissionMode::ReadOnly)
            .with_tool_requirement("write_file", PermissionMode::WorkspaceWrite)
            .with_tool_requirement("bash", PermissionMode::DangerFullAccess);

        assert!(matches!(
            policy.authorize("write_file", "{}", None),
            PermissionOutcome::Deny { reason } if reason.contains("requires workspace-write permission")
        ));
        assert!(matches!(
            policy.authorize("bash", "{}", None),
            PermissionOutcome::Deny { reason } if reason.contains("requires danger-full-access permission")
        ));
    }

    #[test]
    fn prompts_for_workspace_write_to_danger_full_access_escalation() {
        let policy = PermissionPolicy::new(PermissionMode::WorkspaceWrite)
            .with_tool_requirement("bash", PermissionMode::DangerFullAccess);
        let mut prompter = RecordingPrompter {
            seen: Vec::new(),
            allow: true,
        };

        let outcome = policy.authorize("bash", "echo hi", Some(&mut prompter));

        assert_eq!(outcome, PermissionOutcome::Allow);
        assert_eq!(prompter.seen.len(), 1);
        assert_eq!(prompter.seen[0].tool_name, "bash");
        assert_eq!(
            prompter.seen[0].current_mode,
            PermissionMode::WorkspaceWrite
        );
        assert_eq!(
            prompter.seen[0].required_mode,
            PermissionMode::DangerFullAccess
        );
    }

    #[test]
    fn honors_prompt_rejection_reason() {
        let policy = PermissionPolicy::new(PermissionMode::WorkspaceWrite)
            .with_tool_requirement("bash", PermissionMode::DangerFullAccess);
        let mut prompter = RecordingPrompter {
            seen: Vec::new(),
            allow: false,
        };

        assert!(matches!(
            policy.authorize("bash", "echo hi", Some(&mut prompter)),
            PermissionOutcome::Deny { reason } if reason == "not now"
        ));
    }

    #[test]
    fn hook_allow_override_skips_normal_escalation() {
        let policy = PermissionPolicy::new(PermissionMode::ReadOnly)
            .with_tool_requirement("write_file", PermissionMode::WorkspaceWrite);

        assert_eq!(
            policy.authorize_with_context(
                "write_file",
                "{}",
                &PermissionContext::new(Some(PermissionOverride::Allow), None),
                None,
            ),
            PermissionOutcome::Allow
        );
    }

    #[test]
    fn hook_deny_override_returns_hook_reason() {
        let policy = PermissionPolicy::new(PermissionMode::DangerFullAccess);

        assert!(matches!(
            policy.authorize_with_context(
                "bash",
                "{}",
                &PermissionContext::new(
                    Some(PermissionOverride::Deny),
                    Some("blocked by policy hook".to_string()),
                ),
                None,
            ),
            PermissionOutcome::Deny { reason } if reason == "blocked by policy hook"
        ));
    }
}

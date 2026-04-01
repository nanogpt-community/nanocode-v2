mod hooks;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use hooks::{HookEvent, HookRunResult, HookRunner};

const EXTERNAL_MARKETPLACE: &str = "external";
const BUNDLED_MARKETPLACE: &str = "bundled";
const SETTINGS_FILE_NAME: &str = "settings.json";
const REGISTRY_FILE_NAME: &str = "installed.json";
const MANIFEST_FILE_NAME: &str = "plugin.json";
const MANIFEST_DIR_NAME: &str = ".nanocode-plugin";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PluginKind {
    Bundled,
    External,
}

impl PluginKind {
    #[must_use]
    fn marketplace(self) -> &'static str {
        match self {
            Self::Bundled => BUNDLED_MARKETPLACE,
            Self::External => EXTERNAL_MARKETPLACE,
        }
    }
}

impl Display for PluginKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.marketplace())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginMetadata {
    pub id: String,
    pub name: String,
    pub version: String,
    pub description: String,
    pub kind: PluginKind,
    pub source: String,
    pub default_enabled: bool,
    pub root: Option<PathBuf>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginHooks {
    #[serde(rename = "PreToolUse", default)]
    pub pre_tool_use: Vec<String>,
    #[serde(rename = "PostToolUse", default)]
    pub post_tool_use: Vec<String>,
}

impl PluginHooks {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pre_tool_use.is_empty() && self.post_tool_use.is_empty()
    }

    #[must_use]
    pub fn merged_with(&self, other: &Self) -> Self {
        let mut merged = self.clone();
        merged
            .pre_tool_use
            .extend(other.pre_tool_use.iter().cloned());
        merged
            .post_tool_use
            .extend(other.post_tool_use.iter().cloned());
        merged
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginLifecycle {
    #[serde(rename = "Init", default)]
    pub init: Vec<String>,
    #[serde(rename = "Shutdown", default)]
    pub shutdown: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PluginToolPermission {
    ReadOnly,
    WorkspaceWrite,
    DangerFullAccess,
}

impl PluginToolPermission {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::WorkspaceWrite => "workspace-write",
            Self::DangerFullAccess => "danger-full-access",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "read-only" => Some(Self::ReadOnly),
            "workspace-write" => Some(Self::WorkspaceWrite),
            "danger-full-access" => Some(Self::DangerFullAccess),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PluginToolDefinition {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PluginTool {
    plugin_id: String,
    plugin_name: String,
    definition: PluginToolDefinition,
    command: String,
    args: Vec<String>,
    required_permission: PluginToolPermission,
    root: Option<PathBuf>,
}

impl PluginTool {
    #[must_use]
    pub fn new(
        plugin_id: impl Into<String>,
        plugin_name: impl Into<String>,
        definition: PluginToolDefinition,
        command: impl Into<String>,
        args: Vec<String>,
        required_permission: PluginToolPermission,
        root: Option<PathBuf>,
    ) -> Self {
        Self {
            plugin_id: plugin_id.into(),
            plugin_name: plugin_name.into(),
            definition,
            command: command.into(),
            args,
            required_permission,
            root,
        }
    }

    #[must_use]
    pub fn plugin_id(&self) -> &str {
        &self.plugin_id
    }

    #[must_use]
    pub fn definition(&self) -> &PluginToolDefinition {
        &self.definition
    }

    #[must_use]
    pub fn required_permission(&self) -> &str {
        self.required_permission.as_str()
    }

    pub fn execute(&self, input: &Value) -> Result<String, PluginError> {
        let input_json = input.to_string();
        let mut process = Command::new(&self.command);
        process
            .args(&self.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("NANOCODE_PLUGIN_ID", &self.plugin_id)
            .env("NANOCODE_PLUGIN_NAME", &self.plugin_name)
            .env("NANOCODE_TOOL_NAME", &self.definition.name)
            .env("NANOCODE_TOOL_INPUT", &input_json);
        if let Some(root) = &self.root {
            process
                .current_dir(root)
                .env("NANOCODE_PLUGIN_ROOT", root.display().to_string());
        }

        let mut child = process.spawn()?;
        if let Some(stdin) = child.stdin.as_mut() {
            use std::io::Write as _;
            stdin.write_all(input_json.as_bytes())?;
        }

        let output = child.wait_with_output()?;
        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            Err(PluginError::CommandFailed(format!(
                "plugin tool `{}` from `{}` failed for `{}`: {}",
                self.definition.name,
                self.plugin_id,
                self.command,
                if stderr.is_empty() {
                    format!("exit status {}", output.status)
                } else {
                    stderr
                }
            )))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginCommandManifest {
    pub name: String,
    pub description: String,
    pub command: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct RawPluginManifest {
    pub name: String,
    pub version: String,
    pub description: String,
    #[serde(rename = "defaultEnabled", default)]
    pub default_enabled: bool,
    #[serde(default)]
    pub hooks: PluginHooks,
    #[serde(default)]
    pub lifecycle: PluginLifecycle,
    #[serde(default)]
    pub tools: Vec<RawPluginToolManifest>,
    #[serde(default)]
    pub commands: Vec<PluginCommandManifest>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct RawPluginToolManifest {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(
        rename = "requiredPermission",
        default = "default_tool_permission_label"
    )]
    pub required_permission: String,
}

fn default_tool_permission_label() -> String {
    "danger-full-access".to_string()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PluginInstallSource {
    LocalPath { path: PathBuf },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledPluginRecord {
    pub kind: PluginKind,
    pub id: String,
    pub name: String,
    pub version: String,
    pub description: String,
    pub install_path: PathBuf,
    pub source: PluginInstallSource,
    pub installed_at_unix_ms: u128,
    pub updated_at_unix_ms: u128,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledPluginRegistry {
    #[serde(default)]
    pub plugins: BTreeMap<String, InstalledPluginRecord>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RegisteredPlugin {
    metadata: PluginMetadata,
    hooks: PluginHooks,
    lifecycle: PluginLifecycle,
    tools: Vec<PluginTool>,
    enabled: bool,
}

impl RegisteredPlugin {
    #[must_use]
    pub fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    #[must_use]
    pub fn hooks(&self) -> &PluginHooks {
        &self.hooks
    }

    #[must_use]
    pub fn tools(&self) -> &[PluginTool] {
        &self.tools
    }

    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    #[must_use]
    pub fn summary(&self) -> PluginSummary {
        PluginSummary {
            metadata: self.metadata.clone(),
            enabled: self.enabled,
        }
    }

    fn initialize(&self) -> Result<(), PluginError> {
        run_lifecycle_commands(
            self.metadata(),
            "init",
            &self.lifecycle.init,
            self.metadata.root.as_deref(),
        )
    }

    fn shutdown(&self) -> Result<(), PluginError> {
        run_lifecycle_commands(
            self.metadata(),
            "shutdown",
            &self.lifecycle.shutdown,
            self.metadata.root.as_deref(),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginSummary {
    pub metadata: PluginMetadata,
    pub enabled: bool,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct PluginRegistry {
    plugins: Vec<RegisteredPlugin>,
}

impl PluginRegistry {
    #[must_use]
    pub fn new(mut plugins: Vec<RegisteredPlugin>) -> Self {
        plugins.sort_by(|left, right| left.metadata.id.cmp(&right.metadata.id));
        Self { plugins }
    }

    #[must_use]
    pub fn plugins(&self) -> &[RegisteredPlugin] {
        &self.plugins
    }

    #[must_use]
    pub fn summaries(&self) -> Vec<PluginSummary> {
        self.plugins.iter().map(RegisteredPlugin::summary).collect()
    }

    pub fn aggregated_hooks(&self) -> Result<PluginHooks, PluginError> {
        let mut hooks = PluginHooks::default();
        for plugin in self.plugins.iter().filter(|plugin| plugin.enabled) {
            hooks = hooks.merged_with(plugin.hooks());
        }
        Ok(hooks)
    }

    pub fn aggregated_tools(&self) -> Result<Vec<PluginTool>, PluginError> {
        let mut tools = Vec::new();
        let mut seen_names = BTreeMap::new();
        for plugin in self.plugins.iter().filter(|plugin| plugin.enabled) {
            for tool in plugin.tools() {
                if let Some(existing_plugin) =
                    seen_names.insert(tool.definition().name.clone(), tool.plugin_id().to_string())
                {
                    return Err(PluginError::InvalidManifest(format!(
                        "plugin tool `{}` is defined by both `{existing_plugin}` and `{}`",
                        tool.definition().name,
                        tool.plugin_id()
                    )));
                }
                tools.push(tool.clone());
            }
        }
        Ok(tools)
    }

    pub fn initialize(&self) -> Result<(), PluginError> {
        for plugin in self.plugins.iter().filter(|plugin| plugin.enabled) {
            plugin.initialize()?;
        }
        Ok(())
    }

    pub fn shutdown(&self) -> Result<(), PluginError> {
        for plugin in self.plugins.iter().rev().filter(|plugin| plugin.enabled) {
            plugin.shutdown()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginManagerConfig {
    pub config_home: PathBuf,
    pub enabled_plugins: BTreeMap<String, bool>,
    pub external_dirs: Vec<PathBuf>,
    pub install_root: Option<PathBuf>,
    pub registry_path: Option<PathBuf>,
    pub bundled_root: Option<PathBuf>,
}

impl PluginManagerConfig {
    #[must_use]
    pub fn new(config_home: impl Into<PathBuf>) -> Self {
        Self {
            config_home: config_home.into(),
            enabled_plugins: BTreeMap::new(),
            external_dirs: Vec::new(),
            install_root: None,
            registry_path: None,
            bundled_root: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstallOutcome {
    pub plugin_id: String,
    pub version: String,
    pub install_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateOutcome {
    pub plugin_id: String,
    pub old_version: String,
    pub new_version: String,
    pub install_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginManager {
    config: PluginManagerConfig,
}

impl PluginManager {
    #[must_use]
    pub fn new(config: PluginManagerConfig) -> Self {
        Self { config }
    }

    #[must_use]
    pub fn settings_path(&self) -> PathBuf {
        self.config
            .config_home
            .join("plugins")
            .join(SETTINGS_FILE_NAME)
    }

    #[must_use]
    pub fn registry_path(&self) -> PathBuf {
        self.config.registry_path.clone().unwrap_or_else(|| {
            self.config
                .config_home
                .join("plugins")
                .join(REGISTRY_FILE_NAME)
        })
    }

    #[must_use]
    pub fn install_root(&self) -> PathBuf {
        self.config
            .install_root
            .clone()
            .unwrap_or_else(|| self.config.config_home.join("plugins").join("installed"))
    }

    #[must_use]
    pub fn bundled_root(&self) -> Option<PathBuf> {
        self.config.bundled_root.clone()
    }

    pub fn plugin_registry(&self) -> Result<PluginRegistry, PluginError> {
        Ok(PluginRegistry::new(self.load_registered_plugins()?))
    }

    pub fn aggregated_hooks(&self) -> Result<PluginHooks, PluginError> {
        self.plugin_registry()?.aggregated_hooks()
    }

    pub fn aggregated_tools(&self) -> Result<Vec<PluginTool>, PluginError> {
        self.plugin_registry()?.aggregated_tools()
    }

    pub fn list_installed_plugins(&self) -> Result<Vec<PluginSummary>, PluginError> {
        Ok(self
            .load_registered_plugins()?
            .into_iter()
            .map(|plugin| plugin.summary())
            .collect())
    }

    pub fn validate_plugin_source(&self, source: &str) -> Result<PluginMetadata, PluginError> {
        load_plugin_from_directory(&PathBuf::from(source), PluginKind::External)
            .map(|plugin| plugin.metadata)
    }

    pub fn install(&mut self, source: &str) -> Result<InstallOutcome, PluginError> {
        let source_path = PathBuf::from(source);
        let plugin = load_plugin_from_directory(&source_path, PluginKind::External)?;
        let install_path = self.install_root().join(format!(
            "{}-external",
            sanitize_plugin_dir_name(&plugin.metadata.name)
        ));
        if install_path.exists() {
            fs::remove_dir_all(&install_path)?;
        }
        copy_dir_recursive(&source_path, &install_path)?;

        let timestamp = unix_time_ms();
        let mut registry = self.load_registry()?;
        registry.plugins.insert(
            plugin.metadata.id.clone(),
            InstalledPluginRecord {
                kind: PluginKind::External,
                id: plugin.metadata.id.clone(),
                name: plugin.metadata.name.clone(),
                version: plugin.metadata.version.clone(),
                description: plugin.metadata.description.clone(),
                install_path: install_path.clone(),
                source: PluginInstallSource::LocalPath { path: source_path },
                installed_at_unix_ms: timestamp,
                updated_at_unix_ms: timestamp,
            },
        );
        self.store_registry(&registry)?;
        self.write_enabled_state(&plugin.metadata.id, Some(plugin.metadata.default_enabled))?;
        Ok(InstallOutcome {
            plugin_id: plugin.metadata.id,
            version: plugin.metadata.version,
            install_path,
        })
    }

    pub fn update(&mut self, plugin_id: &str) -> Result<UpdateOutcome, PluginError> {
        let mut registry = self.load_registry()?;
        let Some(record) = registry.plugins.get_mut(plugin_id) else {
            return Err(PluginError::NotFound(format!(
                "plugin `{plugin_id}` is not installed"
            )));
        };
        let PluginInstallSource::LocalPath { path } = &record.source;
        let plugin = load_plugin_from_directory(path, record.kind)?;
        if record.install_path.exists() {
            fs::remove_dir_all(&record.install_path)?;
        }
        copy_dir_recursive(path, &record.install_path)?;
        let old_version = record.version.clone();
        let install_path = record.install_path.clone();
        record.version = plugin.metadata.version.clone();
        record.description = plugin.metadata.description.clone();
        record.updated_at_unix_ms = unix_time_ms();
        let new_version = plugin.metadata.version.clone();
        let plugin_id = plugin_id.to_string();
        self.store_registry(&registry)?;
        Ok(UpdateOutcome {
            plugin_id,
            old_version,
            new_version,
            install_path,
        })
    }

    pub fn uninstall(&mut self, plugin_id: &str) -> Result<(), PluginError> {
        let mut registry = self.load_registry()?;
        let Some(record) = registry.plugins.remove(plugin_id) else {
            return Err(PluginError::NotFound(format!(
                "plugin `{plugin_id}` is not installed"
            )));
        };
        if record.install_path.exists() {
            fs::remove_dir_all(&record.install_path)?;
        }
        self.store_registry(&registry)?;
        self.write_enabled_state(plugin_id, None)?;
        Ok(())
    }

    pub fn enable(&mut self, plugin_id: &str) -> Result<(), PluginError> {
        self.ensure_plugin_exists(plugin_id)?;
        self.write_enabled_state(plugin_id, Some(true))
    }

    pub fn disable(&mut self, plugin_id: &str) -> Result<(), PluginError> {
        self.ensure_plugin_exists(plugin_id)?;
        self.write_enabled_state(plugin_id, Some(false))
    }

    pub fn load_registry(&self) -> Result<InstalledPluginRegistry, PluginError> {
        read_optional_json::<InstalledPluginRegistry>(&self.registry_path())
    }

    pub fn store_registry(&self, registry: &InstalledPluginRegistry) -> Result<(), PluginError> {
        write_json_file(&self.registry_path(), registry)
    }

    fn load_registered_plugins(&self) -> Result<Vec<RegisteredPlugin>, PluginError> {
        let mut enabled_state = self.enabled_state()?;
        let mut plugins = Vec::new();
        let mut seen_ids = BTreeSet::new();
        let mut seen_paths = BTreeSet::new();
        let mut stale_registry_ids = Vec::new();

        if let Some(root) = self.bundled_root() {
            collect_plugins_from_root(
                &root,
                PluginKind::Bundled,
                &enabled_state,
                &mut seen_ids,
                &mut seen_paths,
                &mut plugins,
            )?;
        }

        for dir in &self.config.external_dirs {
            collect_plugins_from_root(
                dir,
                PluginKind::External,
                &enabled_state,
                &mut seen_ids,
                &mut seen_paths,
                &mut plugins,
            )?;
        }

        for install_path in discover_plugin_dirs(&self.install_root())? {
            let plugin = load_plugin_from_directory(&install_path, PluginKind::External)?;
            if seen_ids.insert(plugin.metadata.id.clone()) {
                seen_paths.insert(install_path);
                plugins.push(RegisteredPlugin {
                    enabled: resolve_enabled_state(&enabled_state, &plugin.metadata),
                    ..plugin
                });
            }
        }

        let mut registry = self.load_registry()?;
        for record in registry.plugins.values() {
            if seen_paths.contains(&record.install_path) {
                continue;
            }
            if !record.install_path.exists() || !has_manifest(&record.install_path) {
                stale_registry_ids.push(record.id.clone());
                continue;
            }
            let plugin = load_plugin_from_directory(&record.install_path, record.kind)?;
            if seen_ids.insert(plugin.metadata.id.clone()) {
                seen_paths.insert(record.install_path.clone());
                plugins.push(RegisteredPlugin {
                    enabled: resolve_enabled_state(&enabled_state, &plugin.metadata),
                    ..plugin
                });
            }
        }

        if !stale_registry_ids.is_empty() {
            for plugin_id in &stale_registry_ids {
                registry.plugins.remove(plugin_id);
                enabled_state.remove(plugin_id);
            }
            self.store_registry(&registry)?;
            self.store_enabled_state(&enabled_state)?;
        }

        Ok(plugins)
    }

    fn enabled_state(&self) -> Result<BTreeMap<String, bool>, PluginError> {
        let mut enabled = self.config.enabled_plugins.clone();
        enabled.extend(read_optional_json::<BTreeMap<String, bool>>(
            &self.settings_path(),
        )?);
        Ok(enabled)
    }

    fn write_enabled_state(
        &self,
        plugin_id: &str,
        enabled: Option<bool>,
    ) -> Result<(), PluginError> {
        let path = self.settings_path();
        let mut state = read_optional_json::<BTreeMap<String, bool>>(&path)?;
        match enabled {
            Some(enabled) => {
                state.insert(plugin_id.to_string(), enabled);
            }
            None => {
                state.remove(plugin_id);
            }
        }
        self.store_enabled_state(&state)
    }

    fn store_enabled_state(&self, state: &BTreeMap<String, bool>) -> Result<(), PluginError> {
        write_json_file(&self.settings_path(), state)
    }

    fn ensure_plugin_exists(&self, plugin_id: &str) -> Result<(), PluginError> {
        let exists = self
            .load_registered_plugins()?
            .iter()
            .any(|plugin| plugin.metadata.id == plugin_id);
        if exists {
            Ok(())
        } else {
            Err(PluginError::NotFound(format!(
                "plugin `{plugin_id}` was not found"
            )))
        }
    }
}

#[derive(Debug)]
pub enum PluginError {
    Io(std::io::Error),
    Json(serde_json::Error),
    InvalidManifest(String),
    NotFound(String),
    CommandFailed(String),
}

impl Display for PluginError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::Json(error) => write!(f, "{error}"),
            Self::InvalidManifest(error) => write!(f, "{error}"),
            Self::NotFound(error) => write!(f, "{error}"),
            Self::CommandFailed(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for PluginError {}

impl From<std::io::Error> for PluginError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for PluginError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

fn resolve_enabled_state(
    enabled_state: &BTreeMap<String, bool>,
    metadata: &PluginMetadata,
) -> bool {
    enabled_state
        .get(&metadata.id)
        .copied()
        .unwrap_or(metadata.default_enabled)
}

fn collect_plugins_from_root(
    root: &Path,
    kind: PluginKind,
    enabled_state: &BTreeMap<String, bool>,
    seen_ids: &mut BTreeSet<String>,
    seen_paths: &mut BTreeSet<PathBuf>,
    plugins: &mut Vec<RegisteredPlugin>,
) -> Result<(), PluginError> {
    if !root.exists() {
        return Ok(());
    }

    if has_manifest(root) {
        let plugin = load_plugin_from_directory(root, kind)?;
        if seen_ids.insert(plugin.metadata.id.clone()) {
            seen_paths.insert(root.to_path_buf());
            plugins.push(RegisteredPlugin {
                enabled: resolve_enabled_state(enabled_state, &plugin.metadata),
                ..plugin
            });
        }
        return Ok(());
    }

    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() || !has_manifest(&path) {
            continue;
        }
        let plugin = load_plugin_from_directory(&path, kind)?;
        if seen_ids.insert(plugin.metadata.id.clone()) {
            seen_paths.insert(path.clone());
            plugins.push(RegisteredPlugin {
                enabled: resolve_enabled_state(enabled_state, &plugin.metadata),
                ..plugin
            });
        }
    }

    Ok(())
}

fn discover_plugin_dirs(root: &Path) -> Result<Vec<PathBuf>, PluginError> {
    if !root.exists() {
        return Ok(Vec::new());
    }

    if has_manifest(root) {
        return Ok(vec![root.to_path_buf()]);
    }

    let mut dirs = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() && has_manifest(&path) {
            dirs.push(path);
        }
    }
    Ok(dirs)
}

fn has_manifest(root: &Path) -> bool {
    manifest_path(root).is_some()
}

fn manifest_path(root: &Path) -> Option<PathBuf> {
    let packaged = root.join(MANIFEST_DIR_NAME).join(MANIFEST_FILE_NAME);
    if packaged.exists() {
        return Some(packaged);
    }
    let root_manifest = root.join(MANIFEST_FILE_NAME);
    if root_manifest.exists() {
        return Some(root_manifest);
    }
    None
}

fn load_plugin_from_directory(
    root: &Path,
    kind: PluginKind,
) -> Result<RegisteredPlugin, PluginError> {
    let manifest_path = manifest_path(root).ok_or_else(|| {
        PluginError::InvalidManifest(format!(
            "{} does not contain {} or {}",
            root.display(),
            root.join(MANIFEST_DIR_NAME)
                .join(MANIFEST_FILE_NAME)
                .display(),
            root.join(MANIFEST_FILE_NAME).display()
        ))
    })?;
    let manifest = serde_json::from_str::<RawPluginManifest>(&fs::read_to_string(&manifest_path)?)?;
    validate_manifest(&manifest, root)?;
    let plugin_id = format!("{}@{}", manifest.name, kind.marketplace());
    let metadata = PluginMetadata {
        id: plugin_id.clone(),
        name: manifest.name.clone(),
        version: manifest.version.clone(),
        description: manifest.description.clone(),
        kind,
        source: root.display().to_string(),
        default_enabled: manifest.default_enabled,
        root: Some(root.to_path_buf()),
    };
    let tools = manifest
        .tools
        .into_iter()
        .map(|tool| {
            let required_permission =
                PluginToolPermission::parse(&tool.required_permission).ok_or_else(|| {
                    PluginError::InvalidManifest(format!(
                        "plugin tool `{}` requiredPermission `{}` must be read-only, workspace-write, or danger-full-access",
                        tool.name, tool.required_permission
                    ))
                })?;
            Ok(PluginTool::new(
                plugin_id.clone(),
                manifest.name.clone(),
                PluginToolDefinition {
                    name: tool.name,
                    description: Some(tool.description),
                    input_schema: tool.input_schema,
                },
                resolve_command(root, &tool.command),
                tool.args,
                required_permission,
                Some(root.to_path_buf()),
            ))
        })
        .collect::<Result<Vec<_>, PluginError>>()?;
    Ok(RegisteredPlugin {
        metadata,
        hooks: manifest.hooks,
        lifecycle: manifest.lifecycle,
        tools,
        enabled: manifest.default_enabled,
    })
}

fn validate_manifest(manifest: &RawPluginManifest, root: &Path) -> Result<(), PluginError> {
    if manifest.name.trim().is_empty() {
        return Err(PluginError::InvalidManifest(
            "plugin manifest name cannot be empty".to_string(),
        ));
    }
    if manifest.version.trim().is_empty() {
        return Err(PluginError::InvalidManifest(
            "plugin manifest version cannot be empty".to_string(),
        ));
    }
    if manifest.description.trim().is_empty() {
        return Err(PluginError::InvalidManifest(
            "plugin manifest description cannot be empty".to_string(),
        ));
    }

    let mut seen_tools = BTreeSet::new();
    for tool in &manifest.tools {
        if tool.name.trim().is_empty() {
            return Err(PluginError::InvalidManifest(
                "plugin tool name cannot be empty".to_string(),
            ));
        }
        if !seen_tools.insert(tool.name.clone()) {
            return Err(PluginError::InvalidManifest(format!(
                "plugin tool `{}` is duplicated",
                tool.name
            )));
        }
        if !tool.input_schema.is_object() {
            return Err(PluginError::InvalidManifest(format!(
                "plugin tool `{}` inputSchema must be a JSON object",
                tool.name
            )));
        }
        if PluginToolPermission::parse(&tool.required_permission).is_none() {
            return Err(PluginError::InvalidManifest(format!(
                "plugin tool `{}` requiredPermission `{}` must be read-only, workspace-write, or danger-full-access",
                tool.name, tool.required_permission
            )));
        }
        validate_relative_path(root, &tool.command, "tool command")?;
    }

    for command in manifest
        .hooks
        .pre_tool_use
        .iter()
        .chain(manifest.hooks.post_tool_use.iter())
        .chain(manifest.lifecycle.init.iter())
        .chain(manifest.lifecycle.shutdown.iter())
    {
        validate_relative_path(root, command, "plugin script")?;
    }

    Ok(())
}

fn validate_relative_path(root: &Path, value: &str, label: &str) -> Result<(), PluginError> {
    if !is_relative_command_path(value) {
        return Ok(());
    }
    let path = root.join(value);
    if path.exists() {
        Ok(())
    } else {
        Err(PluginError::InvalidManifest(format!(
            "{label} path `{}` does not exist",
            path.display()
        )))
    }
}

fn is_relative_command_path(value: &str) -> bool {
    value.starts_with("./") || value.starts_with("../")
}

fn resolve_command(root: &Path, value: &str) -> String {
    if is_relative_command_path(value) {
        root.join(value).display().to_string()
    } else {
        value.to_string()
    }
}

fn run_lifecycle_commands(
    metadata: &PluginMetadata,
    stage: &str,
    commands: &[String],
    root: Option<&Path>,
) -> Result<(), PluginError> {
    for command in commands {
        let mut process = shell_command(command);
        if let Some(root) = root {
            process.current_dir(root);
        }
        process.env("NANOCODE_PLUGIN_ID", &metadata.id);
        process.env("NANOCODE_PLUGIN_NAME", &metadata.name);
        process.env("NANOCODE_PLUGIN_STAGE", stage);
        let output = process.output()?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            return Err(PluginError::CommandFailed(format!(
                "plugin {stage} command `{command}` failed for `{}`: {}",
                metadata.id,
                if stderr.is_empty() {
                    format!("exit status {}", output.status)
                } else {
                    stderr
                }
            )));
        }
    }
    Ok(())
}

fn shell_command(command: &str) -> Command {
    #[cfg(windows)]
    {
        let mut command_builder = Command::new("cmd");
        command_builder.arg("/C").arg(command);
        command_builder
    }

    #[cfg(not(windows))]
    {
        let mut command_builder = Command::new("sh");
        command_builder.arg("-lc").arg(command);
        command_builder
    }
}

fn sanitize_plugin_dir_name(name: &str) -> String {
    let mut output = name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>();
    while output.contains("--") {
        output = output.replace("--", "-");
    }
    output.trim_matches('-').to_string()
}

fn copy_dir_recursive(source: &Path, destination: &Path) -> Result<(), PluginError> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        if source_path.is_dir() {
            copy_dir_recursive(&source_path, &destination_path)?;
        } else {
            if let Some(parent) = destination_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(&source_path, &destination_path)?;
        }
    }
    Ok(())
}

fn write_json_file<T: Serialize>(path: &Path, value: &T) -> Result<(), PluginError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(value)?)?;
    Ok(())
}

fn read_optional_json<T>(path: &Path) -> Result<T, PluginError>
where
    T: Default + for<'de> Deserialize<'de>,
{
    match fs::read_to_string(path) {
        Ok(contents) if contents.trim().is_empty() => Ok(T::default()),
        Ok(contents) => Ok(serde_json::from_str(&contents)?),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(T::default()),
        Err(error) => Err(PluginError::Io(error)),
    }
}

fn unix_time_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::{
        read_optional_json, HookRunner, PluginKind, PluginManager, PluginManagerConfig,
        PluginTool, PluginToolDefinition, PluginToolPermission, MANIFEST_DIR_NAME,
        MANIFEST_FILE_NAME,
    };
    use serde_json::json;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("nanocode-plugins-{label}-{nanos}"))
    }

    fn write_file(path: &Path, contents: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent dir");
        }
        fs::write(path, contents).expect("write file");
    }

    fn make_executable(path: &Path) {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = fs::metadata(path).expect("metadata").permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(path, permissions).expect("set permissions");
        }
    }

    fn write_plugin(root: &Path, name: &str, version: &str) {
        write_file(
            &root.join("tools").join("echo.sh"),
            "#!/bin/sh\ncat <<'EOF'\n{\"ok\":true}\nEOF\n",
        );
        write_file(
            &root.join(MANIFEST_DIR_NAME).join(MANIFEST_FILE_NAME),
            &format!(
                r#"{{
  "name": "{name}",
  "version": "{version}",
  "description": "demo plugin",
  "defaultEnabled": true,
  "hooks": {{
    "PreToolUse": ["./hooks/pre.sh"]
  }},
  "tools": [
    {{
      "name": "plugin_echo",
      "description": "Echo tool",
      "inputSchema": {{"type":"object"}},
      "command": "./tools/echo.sh",
      "requiredPermission": "workspace-write"
    }}
  ]
}}"#
            ),
        );
        write_file(
            &root.join("hooks").join("pre.sh"),
            "#!/bin/sh\nprintf 'plugin pre'\n",
        );
        make_executable(&root.join("tools").join("echo.sh"));
        make_executable(&root.join("hooks").join("pre.sh"));
    }

    #[test]
    fn installs_lists_and_updates_plugins() {
        let config_home = temp_dir("home");
        let source_root = temp_dir("source");
        write_plugin(&source_root, "demo", "1.0.0");

        let mut manager = PluginManager::new(PluginManagerConfig::new(&config_home));
        let install = manager
            .install(source_root.to_str().expect("utf8 path"))
            .expect("install should succeed");
        assert_eq!(install.plugin_id, "demo@external");

        let list = manager
            .list_installed_plugins()
            .expect("list should succeed");
        assert!(list
            .iter()
            .any(|plugin| plugin.metadata.id == "demo@external"));

        write_plugin(&source_root, "demo", "2.0.0");
        let update = manager
            .update("demo@external")
            .expect("update should succeed");
        assert_eq!(update.old_version, "1.0.0");
        assert_eq!(update.new_version, "2.0.0");

        manager
            .disable("demo@external")
            .expect("disable should succeed");
        let list = manager
            .list_installed_plugins()
            .expect("list should succeed");
        assert!(list
            .iter()
            .any(|plugin| plugin.metadata.id == "demo@external" && !plugin.enabled));

        manager
            .uninstall("demo@external")
            .expect("uninstall should succeed");
        let list = manager
            .list_installed_plugins()
            .expect("list should succeed");
        assert!(!list
            .iter()
            .any(|plugin| plugin.metadata.id == "demo@external"));

        let _ = fs::remove_dir_all(config_home);
        let _ = fs::remove_dir_all(source_root);
    }

    #[test]
    fn aggregates_hooks_and_tools() {
        let config_home = temp_dir("hooks-home");
        let source_root = temp_dir("hooks-source");
        write_plugin(&source_root, "demo", "1.0.0");

        let mut manager = PluginManager::new(PluginManagerConfig::new(&config_home));
        manager
            .install(source_root.to_str().expect("utf8 path"))
            .expect("install should succeed");

        let hooks = manager.aggregated_hooks().expect("hooks should aggregate");
        assert_eq!(hooks.pre_tool_use.len(), 1);

        let tools = manager.aggregated_tools().expect("tools should aggregate");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].definition().name, "plugin_echo");

        let _ = fs::remove_dir_all(config_home);
        let _ = fs::remove_dir_all(source_root);
    }

    #[test]
    fn hook_runner_supports_registry_hooks() {
        let runner = HookRunner::new(crate::PluginHooks {
            pre_tool_use: vec!["printf 'blocked'; exit 2".to_string()],
            post_tool_use: Vec::new(),
        });
        let result = runner.run_pre_tool_use("bash", r#"{"command":"pwd"}"#);
        assert!(result.is_denied());
    }

    #[test]
    fn plugin_tool_executes() {
        let root = temp_dir("tool-root");
        let script = root.join("echo.sh");
        write_file(&script, "#!/bin/sh\ncat\n");
        make_executable(&script);

        let tool = PluginTool::new(
            "demo@external",
            "demo",
            PluginToolDefinition {
                name: "plugin_echo".to_string(),
                description: Some("echo".to_string()),
                input_schema: json!({"type":"object"}),
            },
            script.display().to_string(),
            Vec::new(),
            PluginToolPermission::WorkspaceWrite,
            Some(root.clone()),
        );
        let output = tool
            .execute(&json!({"message":"hello"}))
            .expect("tool should run");
        assert!(output.contains("message"));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn can_load_bundled_plugins_from_configured_root() {
        let config_home = temp_dir("bundled-home");
        let bundled_root = temp_dir("bundled-root");
        write_plugin(&bundled_root.join("sample"), "sample", "0.1.0");

        let mut config = PluginManagerConfig::new(&config_home);
        config.bundled_root = Some(bundled_root.clone());
        let manager = PluginManager::new(config);
        let plugins = manager
            .list_installed_plugins()
            .expect("bundled plugins should list");
        assert!(plugins.iter().any(|plugin| {
            plugin.metadata.id == "sample@bundled" && plugin.metadata.kind == PluginKind::Bundled
        }));

        let _ = fs::remove_dir_all(config_home);
        let _ = fs::remove_dir_all(bundled_root);
    }

    #[test]
    fn installed_plugin_discovery_prunes_stale_registry_and_enabled_state() {
        let config_home = temp_dir("registry-prune-home");
        let missing_install_path = temp_dir("registry-prune-missing");
        let manager = PluginManager::new(PluginManagerConfig::new(&config_home));

        let mut registry = crate::InstalledPluginRegistry::default();
        registry.plugins.insert(
            "stale@external".to_string(),
            crate::InstalledPluginRecord {
                kind: PluginKind::External,
                id: "stale@external".to_string(),
                name: "stale".to_string(),
                version: "1.0.0".to_string(),
                description: "stale plugin".to_string(),
                install_path: missing_install_path.clone(),
                source: crate::PluginInstallSource::LocalPath {
                    path: missing_install_path,
                },
                installed_at_unix_ms: 1,
                updated_at_unix_ms: 1,
            },
        );
        manager.store_registry(&registry).expect("store registry");
        manager
            .write_enabled_state("stale@external", Some(true))
            .expect("store enabled state");

        let installed = manager
            .list_installed_plugins()
            .expect("stale registry entries should be pruned");
        assert!(!installed
            .iter()
            .any(|plugin| plugin.metadata.id == "stale@external"));

        let registry = manager.load_registry().expect("load registry");
        assert!(!registry.plugins.contains_key("stale@external"));
        let enabled = read_optional_json::<std::collections::BTreeMap<String, bool>>(
            &manager.settings_path(),
        )
        .expect("load enabled state");
        assert!(!enabled.contains_key("stale@external"));

        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn list_installed_plugins_scans_install_root_without_registry_entries() {
        let config_home = temp_dir("installed-scan-home");
        let install_root = config_home.join("plugins").join("installed");
        let plugin_root = install_root.join("scan-demo");
        write_plugin(&plugin_root, "scan-demo", "1.0.0");

        let mut config = PluginManagerConfig::new(&config_home);
        config.install_root = Some(install_root);
        let manager = PluginManager::new(config);

        let installed = manager
            .list_installed_plugins()
            .expect("install-root scan should succeed");
        assert!(installed
            .iter()
            .any(|plugin| plugin.metadata.id == "scan-demo@external"));

        let _ = fs::remove_dir_all(config_home);
    }
}

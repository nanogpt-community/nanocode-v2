use std::fs;
use std::path::{Path, PathBuf};

use platform::write_atomic;

const STARTER_SETTINGS_LOCAL_JSON: &str =
    concat!("{\n", "  \"permissionMode\": \"workspace-write\"\n", "}\n",);
const GITIGNORE_COMMENT: &str = "# Pebble local artifacts";
const GITIGNORE_ENTRIES: [&str; 4] = [
    ".pebble/settings.local.json",
    ".pebble/sessions/",
    ".pebble/agents/",
    ".pebble/mcp/",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InitStatus {
    Created,
    Updated,
    Skipped,
}

impl InitStatus {
    #[must_use]
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Created => "created",
            Self::Updated => "updated",
            Self::Skipped => "skipped (already exists)",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct InitArtifact {
    pub(crate) name: &'static str,
    pub(crate) status: InitStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct InitReport {
    pub(crate) project_root: PathBuf,
    pub(crate) artifacts: Vec<InitArtifact>,
}

impl InitReport {
    #[must_use]
    pub(crate) fn render(&self) -> String {
        let mut lines = vec![
            "Init".to_string(),
            format!("  Project          {}", self.project_root.display()),
        ];
        for artifact in &self.artifacts {
            lines.push(format!(
                "  {:<22} {}",
                artifact.name,
                artifact.status.label()
            ));
        }
        lines.push("  Next step        Review and tailor the generated guidance".to_string());
        lines.join("\n")
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
struct RepoDetection {
    rust_workspace: bool,
    rust_root: bool,
    python: bool,
    package_json: bool,
    typescript: bool,
    nextjs: bool,
    react: bool,
    vite: bool,
    nest: bool,
    src_dir: bool,
    tests_dir: bool,
    rust_dir: bool,
}

pub(crate) fn initialize_repo(cwd: &Path) -> Result<InitReport, Box<dyn std::error::Error>> {
    initialize_repo_with_pebble_md(cwd, &render_init_pebble_md(cwd))
}

pub(crate) fn initialize_repo_with_pebble_md(
    cwd: &Path,
    pebble_md_content: &str,
) -> Result<InitReport, Box<dyn std::error::Error>> {
    let mut artifacts = Vec::new();

    let pebble_dir = cwd.join(".pebble");
    artifacts.push(InitArtifact {
        name: ".pebble/",
        status: ensure_dir(&pebble_dir)?,
    });

    let settings_local = pebble_dir.join("settings.local.json");
    artifacts.push(InitArtifact {
        name: ".pebble/settings.local.json",
        status: write_file_if_missing(&settings_local, STARTER_SETTINGS_LOCAL_JSON)?,
    });

    let gitignore = cwd.join(".gitignore");
    artifacts.push(InitArtifact {
        name: ".gitignore",
        status: ensure_gitignore_entries(&gitignore)?,
    });

    let pebble_md = cwd.join("PEBBLE.md");
    artifacts.push(InitArtifact {
        name: "PEBBLE.md",
        status: write_file_if_missing(&pebble_md, pebble_md_content)?,
    });

    Ok(InitReport {
        project_root: cwd.to_path_buf(),
        artifacts,
    })
}

fn ensure_dir(path: &Path) -> Result<InitStatus, std::io::Error> {
    if path.is_dir() {
        return Ok(InitStatus::Skipped);
    }
    fs::create_dir_all(path)?;
    Ok(InitStatus::Created)
}

fn write_file_if_missing(path: &Path, content: &str) -> Result<InitStatus, std::io::Error> {
    if path.exists() {
        return Ok(InitStatus::Skipped);
    }
    write_atomic(path, content)?;
    Ok(InitStatus::Created)
}

fn ensure_gitignore_entries(path: &Path) -> Result<InitStatus, std::io::Error> {
    if !path.exists() {
        let mut lines = vec![GITIGNORE_COMMENT.to_string()];
        lines.extend(GITIGNORE_ENTRIES.iter().map(|entry| (*entry).to_string()));
        write_atomic(path, format!("{}\n", lines.join("\n")))?;
        return Ok(InitStatus::Created);
    }

    let existing = fs::read_to_string(path)?;
    let mut lines = existing.lines().map(ToOwned::to_owned).collect::<Vec<_>>();
    let mut changed = false;

    if !lines.iter().any(|line| line == GITIGNORE_COMMENT) {
        lines.push(GITIGNORE_COMMENT.to_string());
        changed = true;
    }

    for entry in GITIGNORE_ENTRIES {
        if !lines.iter().any(|line| line == entry) {
            lines.push(entry.to_string());
            changed = true;
        }
    }

    if !changed {
        return Ok(InitStatus::Skipped);
    }

    write_atomic(path, format!("{}\n", lines.join("\n")))?;
    Ok(InitStatus::Updated)
}

pub(crate) fn render_init_pebble_md(cwd: &Path) -> String {
    let detection = detect_repo(cwd);
    let mut lines = vec![
        "# PEBBLE.md".to_string(),
        String::new(),
        "This file gives Pebble repo-specific guidance for working in this project.".to_string(),
        String::new(),
    ];

    let detected_languages = detected_languages(&detection);
    let detected_frameworks = detected_frameworks(&detection);
    lines.push("## Detected stack".to_string());
    if detected_languages.is_empty() {
        lines.push("- No specific language markers were detected yet; document the primary language and verification commands once the project structure settles.".to_string());
    } else {
        lines.push(format!("- Languages: {}.", detected_languages.join(", ")));
    }
    if detected_frameworks.is_empty() {
        lines.push("- Frameworks: none detected from the supported starter markers.".to_string());
    } else {
        lines.push(format!(
            "- Frameworks/tooling markers: {}.",
            detected_frameworks.join(", ")
        ));
    }
    lines.push(String::new());

    let verification_lines = verification_lines(cwd, &detection);
    if !verification_lines.is_empty() {
        lines.push("## Verification".to_string());
        lines.extend(verification_lines);
        lines.push(String::new());
    }

    let structure_lines = repository_shape_lines(&detection);
    if !structure_lines.is_empty() {
        lines.push("## Repository shape".to_string());
        lines.extend(structure_lines);
        lines.push(String::new());
    }

    let framework_lines = framework_notes(&detection);
    if !framework_lines.is_empty() {
        lines.push("## Framework notes".to_string());
        lines.extend(framework_lines);
        lines.push(String::new());
    }

    lines.push("## Working agreement".to_string());
    lines.push("- Prefer small, reviewable changes and keep generated bootstrap files aligned with actual repo workflows.".to_string());
    lines.push("- Keep shared defaults in `.pebble/settings.json`; reserve `.pebble/settings.local.json` for machine-local overrides.".to_string());
    lines.push(
        "- Update `PEBBLE.md` intentionally when repo workflows or conventions change.".to_string(),
    );
    lines.push(String::new());

    lines.join("\n")
}

fn detect_repo(cwd: &Path) -> RepoDetection {
    let package_json_contents = fs::read_to_string(cwd.join("package.json"))
        .unwrap_or_default()
        .to_ascii_lowercase();
    RepoDetection {
        rust_workspace: cwd.join("rust").join("Cargo.toml").is_file(),
        rust_root: cwd.join("Cargo.toml").is_file(),
        python: cwd.join("pyproject.toml").is_file()
            || cwd.join("requirements.txt").is_file()
            || cwd.join("setup.py").is_file(),
        package_json: cwd.join("package.json").is_file(),
        typescript: cwd.join("tsconfig.json").is_file()
            || package_json_contents.contains("typescript"),
        nextjs: package_json_contents.contains("\"next\""),
        react: package_json_contents.contains("\"react\""),
        vite: package_json_contents.contains("\"vite\""),
        nest: package_json_contents.contains("@nestjs"),
        src_dir: cwd.join("src").is_dir(),
        tests_dir: cwd.join("tests").is_dir(),
        rust_dir: cwd.join("rust").is_dir(),
    }
}

fn detected_languages(detection: &RepoDetection) -> Vec<&'static str> {
    let mut languages = Vec::new();
    if detection.rust_workspace || detection.rust_root {
        languages.push("Rust");
    }
    if detection.python {
        languages.push("Python");
    }
    if detection.typescript {
        languages.push("TypeScript");
    } else if detection.package_json {
        languages.push("JavaScript/Node.js");
    }
    languages
}

fn detected_frameworks(detection: &RepoDetection) -> Vec<&'static str> {
    let mut frameworks = Vec::new();
    if detection.nextjs {
        frameworks.push("Next.js");
    }
    if detection.react {
        frameworks.push("React");
    }
    if detection.vite {
        frameworks.push("Vite");
    }
    if detection.nest {
        frameworks.push("NestJS");
    }
    frameworks
}

fn verification_lines(cwd: &Path, detection: &RepoDetection) -> Vec<String> {
    let mut lines = Vec::new();
    if detection.rust_root {
        lines.push("- Run Rust verification from the repo root: `cargo fmt --all`, `cargo test --workspace`".to_string());
    }
    if detection.rust_workspace || detection.rust_dir {
        let rust_path = if cwd.join("rust").is_dir() {
            "`rust/`"
        } else {
            "the current directory"
        };
        lines.push(format!(
            "- Run Rust verification from {rust_path}: `cargo fmt --all`, `cargo test --workspace`"
        ));
    }
    if detection.python {
        lines.push(
            "- Document the canonical Python test/lint commands once they are established."
                .to_string(),
        );
    }
    if detection.package_json {
        lines.push("- Document the canonical Node/JS package manager and verification commands once they are established.".to_string());
    }
    lines
}

fn repository_shape_lines(detection: &RepoDetection) -> Vec<String> {
    let mut lines = Vec::new();
    if detection.rust_workspace {
        lines.push("- Nested `rust/` workspace detected.".to_string());
    }
    if detection.rust_root {
        lines.push("- Rust workspace root detected at the current directory.".to_string());
    }
    if detection.src_dir {
        lines.push("- `src/` directory exists.".to_string());
    }
    if detection.tests_dir {
        lines.push("- `tests/` directory exists.".to_string());
    }
    lines
}

fn framework_notes(detection: &RepoDetection) -> Vec<String> {
    let mut lines = Vec::new();
    if detection.nextjs {
        lines
            .push("- Prefer checking route/layout structure before making UI changes.".to_string());
    }
    if detection.react {
        lines.push(
            "- Keep React changes consistent with the existing component/data flow patterns."
                .to_string(),
        );
    }
    if detection.vite {
        lines.push(
            "- Keep Vite project scripts and generated assets out of hand-written source changes."
                .to_string(),
        );
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::{initialize_repo, initialize_repo_with_pebble_md, render_init_pebble_md};

    #[test]
    fn renders_pebble_md_header() {
        let rendered = render_init_pebble_md(std::path::Path::new("."));
        assert!(rendered.contains("# PEBBLE.md"));
        assert!(rendered.contains("Pebble"));
    }

    #[test]
    fn initialize_repo_creates_pebble_artifacts() {
        let root = std::env::temp_dir().join(format!(
            "pebble-init-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time should work")
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).expect("temp root should exist");

        let report = initialize_repo(&root).expect("init should succeed");
        assert!(report
            .artifacts
            .iter()
            .any(|artifact| artifact.name == "PEBBLE.md"));
        assert!(root.join("PEBBLE.md").is_file());
        assert!(root.join(".pebble").is_dir());

        std::fs::remove_dir_all(&root).expect("temp root should be removable");
    }

    #[test]
    fn initialize_repo_writes_custom_pebble_md() {
        let root = std::env::temp_dir().join(format!(
            "pebble-init-custom-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time should work")
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).expect("temp root should exist");

        initialize_repo_with_pebble_md(&root, "# PEBBLE.md\n\nCustom project guidance\n")
            .expect("init should succeed");
        let written = std::fs::read_to_string(root.join("PEBBLE.md")).expect("pebble md exists");
        assert!(written.contains("Custom project guidance"));

        std::fs::remove_dir_all(&root).expect("temp root should be removable");
    }
}

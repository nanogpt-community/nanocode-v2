use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;

use api::{
    ModelCapabilities, ModelInfo, ModelPricing, ModelProvider, NanoGptClient, ProviderPrice,
};
use crossterm::cursor::MoveTo;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, size, Clear, ClearType};
use serde::{Deserialize, Serialize};

const DEFAULT_BASE_URL: &str = "https://nano-gpt.com/api";
const DEFAULT_VISIBLE_ROWS: usize = 14;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelState {
    #[serde(default)]
    pub current_model: Option<String>,
    #[serde(default)]
    pub favorite_models: Vec<String>,
    #[serde(default)]
    pub provider_overrides: BTreeMap<String, String>,
    #[serde(default)]
    pub proxy_tool_calls: bool,
    #[serde(default)]
    pub max_output_tokens_by_model: BTreeMap<String, u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSelection {
    pub selected_model: Option<String>,
    pub favorites_changed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderSelection {
    pub selected_provider: Option<String>,
}

pub fn load_model_state() -> Result<ModelState, Box<dyn std::error::Error>> {
    let path = state_path()?;
    match fs::read_to_string(path) {
        Ok(contents) => Ok(serde_json::from_str(&contents)?),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(ModelState::default()),
        Err(error) => Err(Box::new(error)),
    }
}

pub fn save_model_state(state: &ModelState) -> Result<(), Box<dyn std::error::Error>> {
    let config_home = nanocode_config_home()?;
    fs::create_dir_all(&config_home)?;
    let path = state_path()?;
    fs::write(&path, serde_json::to_string_pretty(state)?)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

pub fn default_model_or(fallback: &str) -> String {
    load_model_state()
        .ok()
        .and_then(|state| state.current_model)
        .unwrap_or_else(|| fallback.to_string())
}

pub fn persist_current_model(model: String) -> Result<(), Box<dyn std::error::Error>> {
    let mut state = load_model_state()?;
    state.current_model = Some(model);
    save_model_state(&state)
}

pub fn provider_for_model(model: &str) -> Option<String> {
    load_model_state()
        .ok()
        .and_then(|state| state.provider_overrides.get(model).cloned())
}

pub fn persist_provider_for_model(
    model: &str,
    provider: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut state = load_model_state()?;
    match provider {
        Some(provider) if !provider.is_empty() => {
            state.provider_overrides.insert(model.to_string(), provider);
        }
        _ => {
            state.provider_overrides.remove(model);
        }
    }
    save_model_state(&state)
}

pub fn proxy_tool_calls_enabled() -> bool {
    load_model_state()
        .map(|state| state.proxy_tool_calls)
        .unwrap_or(false)
}

pub fn max_output_tokens_for_model_or(model: &str, fallback: u32) -> u32 {
    if let Ok(state) = load_model_state() {
        if let Some(value) = state.max_output_tokens_by_model.get(model).copied() {
            return value.max(1);
        }
    }

    let client = build_catalog_client();
    let runtime = match tokio::runtime::Runtime::new() {
        Ok(runtime) => runtime,
        Err(_) => return fallback,
    };
    let models = match runtime.block_on(client.fetch_models(true)) {
        Ok(response) => response.data,
        Err(_) => return fallback,
    };

    let mut state = load_model_state().unwrap_or_default();
    update_output_token_cache(&mut state, &models);
    let resolved = state
        .max_output_tokens_by_model
        .get(model)
        .copied()
        .filter(|value| *value > 0)
        .unwrap_or(fallback);
    let _ = save_model_state(&state);
    resolved
}

pub fn persist_proxy_tool_calls(enabled: bool) -> Result<(), Box<dyn std::error::Error>> {
    let mut state = load_model_state()?;
    state.proxy_tool_calls = enabled;
    save_model_state(&state)
}

pub fn open_model_picker() -> Result<ModelSelection, Box<dyn std::error::Error>> {
    let client = build_catalog_client();
    let mut state = load_model_state()?;
    let models = fetch_sorted_models(&client, &state)?;
    update_output_token_cache(&mut state, &models);
    if models.is_empty() {
        return Err("NanoGPT returned an empty model list".into());
    }

    let selection = interactive_model_picker(&models, &mut state)?;
    if let Some(model) = &selection.selected_model {
        state.current_model = Some(model.clone());
    }
    save_model_state(&state)?;
    Ok(selection)
}

pub fn open_provider_picker(model: &str) -> Result<ProviderSelection, Box<dyn std::error::Error>> {
    let response = fetch_provider_selection(model)?;
    if !response.supports_provider_selection {
        return Err(format!("provider selection is not supported for {model}").into());
    }

    let mut state = load_model_state()?;
    let selected_provider = interactive_provider_picker(
        model,
        response.default_price.as_ref(),
        &response.providers,
        state.provider_overrides.get(model).cloned(),
    )?;
    match &selected_provider.selected_provider {
        Some(provider) => {
            state
                .provider_overrides
                .insert(model.to_string(), provider.clone());
        }
        None => {
            state.provider_overrides.remove(model);
        }
    }
    save_model_state(&state)?;
    Ok(selected_provider)
}

pub fn validate_provider_for_model(
    model: &str,
    provider: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let response = fetch_provider_selection(model)?;
    if !response.supports_provider_selection {
        return Err(format!("provider selection is not supported for {model}").into());
    }

    let provider_entry = response
        .providers
        .iter()
        .find(|entry| entry.provider == provider)
        .ok_or_else(|| format!("unknown provider for {model}: {provider}"))?;
    if provider_entry.available == Some(false) {
        return Err(format!("provider is currently unavailable for {model}: {provider}").into());
    }
    Ok(())
}

fn build_catalog_client() -> NanoGptClient {
    let api_key = std::env::var("NANOGPT_API_KEY")
        .ok()
        .filter(|value| !value.is_empty())
        .or_else(|| load_credentials_api_key().ok().flatten())
        .unwrap_or_default();
    NanoGptClient::new(api_key).with_base_url(
        std::env::var("NANOGPT_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string()),
    )
}

fn fetch_provider_selection(
    model: &str,
) -> Result<api::ProviderSelectionResponse, Box<dyn std::error::Error>> {
    let client = build_catalog_client();
    let runtime = tokio::runtime::Runtime::new()?;
    Ok(runtime.block_on(client.fetch_providers(model))?)
}

fn fetch_sorted_models(
    client: &NanoGptClient,
    state: &ModelState,
) -> Result<Vec<ModelInfo>, Box<dyn std::error::Error>> {
    let runtime = tokio::runtime::Runtime::new()?;
    let mut models = runtime.block_on(client.fetch_models(true))?.data;
    models.sort_by(|left, right| compare_models(left, right, state));
    Ok(models)
}

fn update_output_token_cache(state: &mut ModelState, models: &[ModelInfo]) {
    for model in models {
        if let Some(max_output_tokens) = model
            .max_output_tokens
            .map(|value| value.min(u64::from(u32::MAX)) as u32)
            .filter(|value| *value > 0)
        {
            state
                .max_output_tokens_by_model
                .insert(model.id.clone(), max_output_tokens);
        }
    }
}

fn compare_models(left: &ModelInfo, right: &ModelInfo, state: &ModelState) -> Ordering {
    let left_favorite = state.favorite_models.iter().any(|item| item == &left.id);
    let right_favorite = state.favorite_models.iter().any(|item| item == &right.id);
    let left_current = state.current_model.as_deref() == Some(left.id.as_str());
    let right_current = state.current_model.as_deref() == Some(right.id.as_str());

    right_current
        .cmp(&left_current)
        .then_with(|| right_favorite.cmp(&left_favorite))
        .then_with(|| {
            display_name(left)
                .to_ascii_lowercase()
                .cmp(&display_name(right).to_ascii_lowercase())
        })
        .then_with(|| left.id.cmp(&right.id))
}

fn interactive_model_picker(
    models: &[ModelInfo],
    state: &mut ModelState,
) -> Result<ModelSelection, Box<dyn std::error::Error>> {
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        return select_model_fallback(models, state);
    }

    let current_index = state
        .current_model
        .as_deref()
        .and_then(|model| models.iter().position(|entry| entry.id == model))
        .unwrap_or(0);
    let mut query = String::new();
    let mut search_mode = false;
    let mut filtered_indices = filtered_model_indices(models, &query);
    let mut cursor = filtered_indices
        .iter()
        .position(|index| *index == current_index)
        .unwrap_or(0);
    let mut favorites_changed = false;
    enable_raw_mode()?;
    let mut stdout = io::stdout();

    loop {
        draw_model_picker(
            &mut stdout,
            models,
            state,
            &filtered_indices,
            cursor,
            &query,
            search_mode,
        )?;
        match event::read()? {
            Event::Key(KeyEvent {
                code: KeyCode::Char('/'),
                modifiers: KeyModifiers::NONE,
                ..
            }) => {
                search_mode = true;
            }
            Event::Key(KeyEvent {
                code: KeyCode::Up, ..
            })
            | Event::Key(KeyEvent {
                code: KeyCode::Char('k'),
                modifiers: KeyModifiers::NONE,
                ..
            }) if !search_mode => {
                cursor = cursor.saturating_sub(1);
            }
            Event::Key(KeyEvent {
                code: KeyCode::Char('k'),
                modifiers: KeyModifiers::NONE,
                ..
            }) => {}
            Event::Key(KeyEvent {
                code: KeyCode::Down,
                ..
            })
            | Event::Key(KeyEvent {
                code: KeyCode::Char('j'),
                modifiers: KeyModifiers::NONE,
                ..
            }) if !search_mode => {
                cursor = (cursor + 1).min(filtered_indices.len().saturating_sub(1));
            }
            Event::Key(KeyEvent {
                code: KeyCode::Char('j'),
                modifiers: KeyModifiers::NONE,
                ..
            }) => {}
            Event::Key(KeyEvent {
                code: KeyCode::PageUp,
                ..
            }) => {
                cursor = cursor.saturating_sub(DEFAULT_VISIBLE_ROWS);
            }
            Event::Key(KeyEvent {
                code: KeyCode::PageDown,
                ..
            }) => {
                cursor =
                    (cursor + DEFAULT_VISIBLE_ROWS).min(filtered_indices.len().saturating_sub(1));
            }
            Event::Key(KeyEvent {
                code: KeyCode::Home,
                ..
            }) => cursor = 0,
            Event::Key(KeyEvent {
                code: KeyCode::End, ..
            }) => {
                cursor = filtered_indices.len().saturating_sub(1);
            }
            Event::Key(KeyEvent {
                code: KeyCode::Char('f'),
                modifiers: KeyModifiers::NONE,
                ..
            }) if !search_mode => {
                if let Some(model_index) = filtered_indices.get(cursor) {
                    toggle_favorite(&models[*model_index].id, state);
                    favorites_changed = true;
                }
            }
            Event::Key(KeyEvent {
                code: KeyCode::Enter,
                ..
            }) => {
                if search_mode {
                    search_mode = false;
                    continue;
                }
                let Some(model_index) = filtered_indices.get(cursor) else {
                    continue;
                };
                disable_raw_mode()?;
                write!(stdout, "\r\n")?;
                return Ok(ModelSelection {
                    selected_model: Some(models[*model_index].id.clone()),
                    favorites_changed,
                });
            }
            Event::Key(KeyEvent {
                code: KeyCode::Backspace,
                ..
            }) => {
                if !query.is_empty() {
                    query.pop();
                    filtered_indices = filtered_model_indices(models, &query);
                    cursor = updated_cursor(&filtered_indices, cursor, current_index);
                    search_mode = true;
                }
            }
            Event::Key(KeyEvent {
                code: KeyCode::Char('u'),
                modifiers,
                ..
            }) if modifiers.contains(KeyModifiers::CONTROL) => {
                query.clear();
                filtered_indices = filtered_model_indices(models, &query);
                cursor = updated_cursor(&filtered_indices, cursor, current_index);
                search_mode = false;
            }
            Event::Key(KeyEvent {
                code: KeyCode::Char(ch),
                modifiers,
                ..
            }) if search_mode && (modifiers.is_empty() || modifiers == KeyModifiers::SHIFT) => {
                query.push(ch);
                filtered_indices = filtered_model_indices(models, &query);
                cursor = updated_cursor(&filtered_indices, cursor, current_index);
            }
            Event::Key(KeyEvent {
                code: KeyCode::Esc, ..
            })
            | Event::Key(KeyEvent {
                code: KeyCode::Char('q'),
                modifiers: KeyModifiers::NONE,
                ..
            }) => {
                if search_mode || !query.is_empty() {
                    query.clear();
                    filtered_indices = filtered_model_indices(models, &query);
                    cursor = updated_cursor(&filtered_indices, cursor, current_index);
                    search_mode = false;
                    continue;
                }
                disable_raw_mode()?;
                write!(stdout, "\r\n")?;
                return Ok(ModelSelection {
                    selected_model: None,
                    favorites_changed,
                });
            }
            Event::Resize(_, _) => {}
            _ => {}
        }
    }
}

fn select_model_fallback(
    models: &[ModelInfo],
    state: &mut ModelState,
) -> Result<ModelSelection, Box<dyn std::error::Error>> {
    println!("NanoGPT Models");
    for (index, model) in models.iter().take(25).enumerate() {
        let current = if state.current_model.as_deref() == Some(model.id.as_str()) {
            ">"
        } else {
            " "
        };
        let favorite = if state.favorite_models.iter().any(|entry| entry == &model.id) {
            "*"
        } else {
            " "
        };
        println!(
            "{:>2}. {}{} {} ({})",
            index + 1,
            current,
            favorite,
            display_name(model),
            model.id
        );
    }
    print!("Choose a model by number or exact id: ");
    io::stdout().flush()?;

    let mut buffer = String::new();
    io::stdin().read_line(&mut buffer)?;
    let input = buffer.trim();
    if input.is_empty() {
        return Ok(ModelSelection {
            selected_model: None,
            favorites_changed: false,
        });
    }

    let selected_model = if let Ok(index) = input.parse::<usize>() {
        models
            .get(index.saturating_sub(1))
            .map(|model| model.id.clone())
            .ok_or_else(|| format!("model number {index} is out of range"))?
    } else {
        models
            .iter()
            .find(|model| model.id == input)
            .map(|model| model.id.clone())
            .ok_or_else(|| format!("unknown model id: {input}"))?
    };

    Ok(ModelSelection {
        selected_model: Some(selected_model),
        favorites_changed: false,
    })
}

fn interactive_provider_picker(
    model: &str,
    default_price: Option<&ProviderPrice>,
    providers: &[ModelProvider],
    current_provider: Option<String>,
) -> Result<ProviderSelection, Box<dyn std::error::Error>> {
    let mut entries = Vec::with_capacity(providers.len() + 1);
    entries.push((
        None,
        format!(
            "Platform default ({})",
            provider_price_summary(default_price)
        ),
    ));
    for provider in providers {
        let availability = if provider.available == Some(false) {
            "unavailable"
        } else {
            "available"
        };
        entries.push((
            Some(provider.provider.clone()),
            format!(
                "{} [{}] {}",
                provider.provider,
                availability,
                provider_price_summary(provider.pricing.as_ref())
            ),
        ));
    }

    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        println!("Providers for {model}");
        for (index, (provider, label)) in entries.iter().enumerate() {
            let current = if current_provider.as_deref() == provider.as_deref() {
                ">"
            } else {
                " "
            };
            println!("{:>2}. {} {}", index + 1, current, label);
        }
        print!("Choose a provider number, exact id, or press Enter for platform default: ");
        io::stdout().flush()?;

        let mut buffer = String::new();
        io::stdin().read_line(&mut buffer)?;
        let input = buffer.trim();
        if input.is_empty() {
            return Ok(ProviderSelection {
                selected_provider: None,
            });
        }

        let selected_provider = if let Ok(index) = input.parse::<usize>() {
            entries
                .get(index.saturating_sub(1))
                .map(|entry| entry.0.clone())
                .ok_or_else(|| format!("provider number {index} is out of range"))?
        } else {
            providers
                .iter()
                .find(|provider| provider.provider == input)
                .map(|provider| Some(provider.provider.clone()))
                .ok_or_else(|| format!("unknown provider id: {input}"))?
        };

        return Ok(ProviderSelection { selected_provider });
    }

    let mut cursor = current_provider
        .as_deref()
        .and_then(|provider| {
            entries
                .iter()
                .position(|entry| entry.0.as_deref() == Some(provider))
        })
        .unwrap_or(0);
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    loop {
        execute!(stdout, MoveTo(0, 0), Clear(ClearType::All))?;
        write_raw_line(&mut stdout, &format!("Providers for {model}"))?;
        write_raw_line(
            &mut stdout,
            "Use Up/Down or j/k to move, Enter to select, q to cancel",
        )?;
        write_raw_line(
            &mut stdout,
            &format!(
                "Current provider: {}",
                current_provider.as_deref().unwrap_or("<platform default>")
            ),
        )?;
        write_raw_line(&mut stdout, "")?;

        for (index, (_, label)) in entries.iter().enumerate() {
            let selected_marker = if index == cursor { ">" } else { " " };
            let current_marker = if current_provider.as_deref() == entries[index].0.as_deref() {
                "o"
            } else {
                " "
            };
            write_raw_line(
                &mut stdout,
                &format!("{}{} {}", selected_marker, current_marker, label),
            )?;
        }
        stdout.flush()?;

        match event::read()? {
            Event::Key(KeyEvent {
                code: KeyCode::Up, ..
            })
            | Event::Key(KeyEvent {
                code: KeyCode::Char('k'),
                modifiers: KeyModifiers::NONE,
                ..
            }) => cursor = cursor.saturating_sub(1),
            Event::Key(KeyEvent {
                code: KeyCode::Down,
                ..
            })
            | Event::Key(KeyEvent {
                code: KeyCode::Char('j'),
                modifiers: KeyModifiers::NONE,
                ..
            }) => cursor = (cursor + 1).min(entries.len().saturating_sub(1)),
            Event::Key(KeyEvent {
                code: KeyCode::Enter,
                ..
            }) => {
                disable_raw_mode()?;
                write!(stdout, "\r\n")?;
                return Ok(ProviderSelection {
                    selected_provider: entries[cursor].0.clone(),
                });
            }
            Event::Key(KeyEvent {
                code: KeyCode::Esc, ..
            })
            | Event::Key(KeyEvent {
                code: KeyCode::Char('q'),
                modifiers: KeyModifiers::NONE,
                ..
            }) => {
                disable_raw_mode()?;
                write!(stdout, "\r\n")?;
                return Ok(ProviderSelection {
                    selected_provider: current_provider,
                });
            }
            _ => {}
        }
    }
}

fn draw_model_picker(
    stdout: &mut impl Write,
    models: &[ModelInfo],
    state: &ModelState,
    filtered_indices: &[usize],
    cursor: usize,
    query: &str,
    search_mode: bool,
) -> io::Result<()> {
    execute!(stdout, MoveTo(0, 0), Clear(ClearType::All))?;
    let (width, height) = terminal_dimensions();
    let content_width = width.saturating_sub(1).clamp(24, 88);
    let visible_rows = model_visible_rows(height);
    write_raw_line(stdout, "NanoGPT Models")?;
    write_raw_line(
        stdout,
        &fit_line(
            "Up/Down move, Enter select, / search, f favorite, q cancel",
            content_width,
        ),
    )?;
    write_raw_line(
        stdout,
        &format!(
            "Current: {}",
            fit_line(
                state.current_model.as_deref().unwrap_or("<unset>"),
                content_width.saturating_sub(9)
            )
        ),
    )?;
    let search_label = if query.is_empty() {
        if search_mode {
            "Search: ".to_string()
        } else {
            "Search: / to filter".to_string()
        }
    } else if search_mode {
        format!("Search: {query}_")
    } else {
        format!("Search: {query}")
    };
    write_raw_line(stdout, &fit_line(&search_label, content_width))?;
    write_raw_line(stdout, "")?;

    let start = cursor.saturating_sub(visible_rows / 2);
    let end = (start + visible_rows).min(filtered_indices.len());
    let start = end.saturating_sub(visible_rows);

    for (visible_index, model_index) in filtered_indices
        .iter()
        .enumerate()
        .skip(start)
        .take(end - start)
    {
        let model = &models[*model_index];
        let selected_marker = if visible_index == cursor { ">" } else { " " };
        let favorite_marker = if state.favorite_models.iter().any(|entry| entry == &model.id) {
            "*"
        } else {
            " "
        };
        let current_marker = if state.current_model.as_deref() == Some(model.id.as_str()) {
            "o"
        } else {
            " "
        };
        write_raw_line(
            stdout,
            &format!(
                "{}{}{} {}",
                selected_marker,
                favorite_marker,
                current_marker,
                fit_line(&model_list_label(model), content_width.saturating_sub(4))
            ),
        )?;
    }

    write_raw_line(stdout, "")?;
    if let Some(model_index) = filtered_indices.get(cursor) {
        let selected = &models[*model_index];
        write_raw_line(
            stdout,
            &fit_line(&selected_summary(selected), content_width),
        )?;
        if let Some(description) = &selected.description {
            for line in wrap_text(description, content_width, 2) {
                write_raw_line(stdout, &line)?;
            }
        }
        for line in wrap_text(&detail_line(selected), content_width, 2) {
            write_raw_line(stdout, &line)?;
        }
    } else {
        write_raw_line(stdout, "No models match the current search.")?;
        write_raw_line(
            stdout,
            "Press / to edit the query, Backspace to delete, Esc to clear.",
        )?;
    }
    stdout.flush()
}

fn terminal_dimensions() -> (usize, usize) {
    size()
        .map(|(width, height)| (usize::from(width), usize::from(height)))
        .unwrap_or((120, 32))
}

fn model_visible_rows(height: usize) -> usize {
    height.saturating_sub(10).clamp(6, DEFAULT_VISIBLE_ROWS)
}

fn model_list_label(model: &ModelInfo) -> String {
    let mut segments = vec![display_name(model).to_string()];
    if display_name(model) != model.id {
        segments.push(short_model_id(&model.id));
    }
    segments.push(model.owned_by.clone());
    if let Some(category) = &model.category {
        segments.push(category.clone());
    }
    segments.join(" | ")
}

fn selected_summary(model: &ModelInfo) -> String {
    format!("Selected: {} | {}", display_name(model), model.id)
}

fn detail_line(model: &ModelInfo) -> String {
    let mut parts = Vec::new();
    if let Some(category) = &model.category {
        parts.push(format!("category={category}"));
    }
    if let Some(context_length) = model.context_length {
        parts.push(format!("ctx={context_length}"));
    }
    if let Some(max_output_tokens) = model.max_output_tokens {
        parts.push(format!("max_out={max_output_tokens}"));
    }
    let pricing = pricing_summary(model.pricing.as_ref());
    if pricing != "pricing n/a" {
        parts.push(pricing);
    }
    let capabilities = capability_summary(model.capabilities.as_ref());
    if !capabilities.is_empty() {
        parts.push(format!("caps={capabilities}"));
    }
    parts.join(" | ")
}

fn capability_summary(capabilities: Option<&ModelCapabilities>) -> String {
    let Some(capabilities) = capabilities else {
        return String::new();
    };
    let mut names = Vec::new();
    if capabilities.vision == Some(true) {
        names.push("vision");
    }
    if capabilities.reasoning == Some(true) {
        names.push("reasoning");
    }
    if capabilities.tool_calling == Some(true) {
        names.push("tools");
    }
    if capabilities.parallel_tool_calls == Some(true) {
        names.push("parallel-tools");
    }
    if capabilities.structured_output == Some(true) {
        names.push("structured-output");
    }
    if capabilities.pdf_upload == Some(true) {
        names.push("pdf");
    }
    names.join(", ")
}

fn pricing_summary(pricing: Option<&ModelPricing>) -> String {
    let Some(pricing) = pricing else {
        return "pricing n/a".to_string();
    };
    match (pricing.prompt, pricing.completion) {
        (Some(prompt), Some(completion)) => format!("in ${prompt:.4}/M out ${completion:.4}/M"),
        (Some(prompt), None) => format!("in ${prompt:.4}/M"),
        (None, Some(completion)) => format!("out ${completion:.4}/M"),
        (None, None) => "pricing n/a".to_string(),
    }
}

fn provider_price_summary(pricing: Option<&ProviderPrice>) -> String {
    let Some(pricing) = pricing else {
        return "pricing n/a".to_string();
    };
    match (pricing.input_per_1k_tokens, pricing.output_per_1k_tokens) {
        (Some(input), Some(output)) => format!("in ${input:.6}/1K out ${output:.6}/1K"),
        (Some(input), None) => format!("in ${input:.6}/1K"),
        (None, Some(output)) => format!("out ${output:.6}/1K"),
        (None, None) => "pricing n/a".to_string(),
    }
}

fn display_name(model: &ModelInfo) -> &str {
    model.name.as_deref().unwrap_or(&model.id)
}

fn short_model_id(model_id: &str) -> String {
    const KEEP: usize = 28;
    if model_id.chars().count() <= KEEP {
        return model_id.to_string();
    }
    truncate(model_id, KEEP)
}

fn write_raw_line(stdout: &mut impl Write, line: &str) -> io::Result<()> {
    write!(stdout, "{line}\r\n")
}

fn filtered_model_indices(models: &[ModelInfo], query: &str) -> Vec<usize> {
    if query.trim().is_empty() {
        return (0..models.len()).collect();
    }

    let query = query.to_ascii_lowercase();
    models
        .iter()
        .enumerate()
        .filter_map(|(index, model)| model_matches_query(model, &query).then_some(index))
        .collect()
}

fn model_matches_query(model: &ModelInfo, query: &str) -> bool {
    [
        Some(display_name(model)),
        Some(model.id.as_str()),
        Some(model.owned_by.as_str()),
        model.category.as_deref(),
        model.description.as_deref(),
    ]
    .into_iter()
    .flatten()
    .any(|value| value.to_ascii_lowercase().contains(query))
}

fn updated_cursor(
    filtered_indices: &[usize],
    previous_cursor: usize,
    fallback_index: usize,
) -> usize {
    if filtered_indices.is_empty() {
        return 0;
    }
    if previous_cursor < filtered_indices.len() {
        return previous_cursor;
    }

    filtered_indices
        .iter()
        .position(|model_index| *model_index == fallback_index)
        .unwrap_or(0)
}

fn truncate(input: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let mut output = String::new();
    let mut count = 0;
    for ch in input.chars() {
        if count >= max_chars {
            if max_chars <= 3 {
                return ".".repeat(max_chars);
            }
            output.truncate(max_chars - 3);
            output.push_str("...");
            return output;
        }
        output.push(ch);
        count += 1;
    }
    output
}

fn fit_line(input: &str, width: usize) -> String {
    truncate(input, width)
}

fn wrap_text(input: &str, width: usize, max_lines: usize) -> Vec<String> {
    if max_lines == 0 {
        return Vec::new();
    }
    let width = width.max(8);
    let mut lines = Vec::new();
    let mut current = String::new();

    for word in input.split_whitespace() {
        let candidate_len = if current.is_empty() {
            word.chars().count()
        } else {
            current.chars().count() + 1 + word.chars().count()
        };
        if candidate_len <= width {
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(word);
            continue;
        }

        if !current.is_empty() {
            lines.push(current);
            if lines.len() == max_lines {
                let last = lines.pop().unwrap_or_default();
                lines.push(truncate(&last, width));
                return lines;
            }
            current = String::new();
        }

        if word.chars().count() > width {
            lines.push(truncate(word, width));
            if lines.len() == max_lines {
                return lines;
            }
        } else {
            current.push_str(word);
        }
    }

    if !current.is_empty() && lines.len() < max_lines {
        lines.push(current);
    }

    lines
}

fn toggle_favorite(model_id: &str, state: &mut ModelState) {
    if let Some(index) = state
        .favorite_models
        .iter()
        .position(|entry| entry == model_id)
    {
        state.favorite_models.remove(index);
    } else {
        state.favorite_models.push(model_id.to_string());
        state.favorite_models.sort();
        state.favorite_models.dedup();
    }
}

fn load_credentials_api_key() -> Result<Option<String>, Box<dyn std::error::Error>> {
    let path = nanocode_config_home()?.join("credentials.json");
    match fs::read_to_string(path) {
        Ok(contents) => {
            let parsed = serde_json::from_str::<serde_json::Value>(&contents)?;
            Ok(parsed
                .get("nanogpt_api_key")
                .and_then(serde_json::Value::as_str)
                .or_else(|| parsed.get("apiKey").and_then(serde_json::Value::as_str))
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned))
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(Box::new(error)),
    }
}

fn state_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(nanocode_config_home()?.join("state.json"))
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

#[cfg(test)]
mod tests {
    use api::ModelInfo;

    use super::{
        filtered_model_indices, toggle_favorite, update_output_token_cache, ModelState,
    };

    #[test]
    fn toggles_favorite_membership() {
        let mut state = ModelState::default();
        toggle_favorite("openai/gpt-5.2", &mut state);
        assert_eq!(state.favorite_models, vec!["openai/gpt-5.2"]);
        toggle_favorite("openai/gpt-5.2", &mut state);
        assert!(state.favorite_models.is_empty());
    }

    #[test]
    fn filters_models_by_name_or_id_or_category() {
        let models = vec![
            model("zai-org/glm-5.1", Some("GLM 5.1"), Some("More")),
            model("openai/gpt-5.4", Some("GPT 5.4"), Some("Flagship")),
        ];

        assert_eq!(filtered_model_indices(&models, "glm"), vec![0]);
        assert_eq!(filtered_model_indices(&models, "gpt-5.4"), vec![1]);
        assert_eq!(filtered_model_indices(&models, "flag"), vec![1]);
    }

    #[test]
    fn caches_max_output_tokens_from_catalog_models() {
        let mut state = ModelState::default();
        let mut models = vec![model("zai-org/glm-5.1", Some("GLM 5.1"), Some("More"))];
        models[0].max_output_tokens = Some(131_072);

        update_output_token_cache(&mut state, &models);

        assert_eq!(
            state.max_output_tokens_by_model.get("zai-org/glm-5.1"),
            Some(&131_072)
        );
    }

    fn model(id: &str, name: Option<&str>, category: Option<&str>) -> ModelInfo {
        ModelInfo {
            id: id.to_string(),
            object: "model".to_string(),
            created: 0,
            owned_by: "provider".to_string(),
            name: name.map(ToOwned::to_owned),
            description: None,
            context_length: None,
            max_output_tokens: None,
            pricing: None,
            capabilities: None,
            category: category.map(ToOwned::to_owned),
            cost_estimate: None,
            tags: None,
            supports_provider_selection: None,
        }
    }
}

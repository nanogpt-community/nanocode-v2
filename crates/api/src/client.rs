use std::collections::VecDeque;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use platform::pebble_config_home;
use serde::Deserialize;
use serde::Serialize;

use crate::error::ApiError;
use crate::sse::SseParser;
use crate::types::{
    ChatCompletionContent, ChatCompletionFunction, ChatCompletionMessage, ChatCompletionRequest,
    ChatCompletionResponse, ChatCompletionThinkingConfig, ChatCompletionTool, MessageRequest,
    MessageResponse, ModelsResponse, OutputContentBlock, ProviderSelectionResponse, StreamEvent,
};

const DEFAULT_BASE_URL: &str = "https://nano-gpt.com/api";
const DEFAULT_SYNTHETIC_MESSAGES_BASE_URL: &str = "https://api.synthetic.new/anthropic/v1";
const DEFAULT_OPENCODE_GO_BASE_URL: &str = "https://opencode.ai/zen/go";
const REQUEST_ID_HEADER: &str = "request-id";
const ALT_REQUEST_ID_HEADER: &str = "x-request-id";
const DEFAULT_INITIAL_BACKOFF: Duration = Duration::from_millis(200);
const DEFAULT_MAX_BACKOFF: Duration = Duration::from_secs(2);
const DEFAULT_MAX_RETRIES: u32 = 2;

fn nanogpt_client_debug_enabled() -> bool {
    std::env::var("NANOGPT_CLIENT_DEBUG")
        .ok()
        .is_some_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on" | "debug"
            )
        })
}

#[derive(Debug, Clone)]
pub struct NanoGptClient {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    service: ApiService,
    provider: Option<String>,
    force_paygo: bool,
    max_retries: u32,
    initial_backoff: Duration,
    max_backoff: Duration,
}

impl NanoGptClient {
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            service: ApiService::NanoGpt,
            provider: None,
            force_paygo: false,
            max_retries: DEFAULT_MAX_RETRIES,
            initial_backoff: DEFAULT_INITIAL_BACKOFF,
            max_backoff: DEFAULT_MAX_BACKOFF,
        }
    }

    pub fn from_env() -> Result<Self, ApiError> {
        Self::from_service_env(ApiService::NanoGpt)
    }

    pub fn from_service_env(service: ApiService) -> Result<Self, ApiError> {
        Ok(Self::new(resolve_api_key_for(service)?)
            .with_service(service)
            .with_base_url(resolve_base_url_for(service)))
    }

    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    #[must_use]
    pub fn with_service(mut self, service: ApiService) -> Self {
        self.service = service;
        self
    }

    #[must_use]
    pub fn with_provider(mut self, provider: Option<String>) -> Self {
        self.provider = provider.filter(|value| !value.is_empty());
        self.force_paygo = self.provider.is_some();
        self
    }

    #[must_use]
    pub fn with_retry_policy(
        mut self,
        max_retries: u32,
        initial_backoff: Duration,
        max_backoff: Duration,
    ) -> Self {
        self.max_retries = max_retries;
        self.initial_backoff = initial_backoff;
        self.max_backoff = max_backoff;
        self
    }

    pub async fn send_message(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageResponse, ApiError> {
        if self.service == ApiService::OpencodeGo && !opencode_go_uses_messages_api(&request.model)
        {
            return self.send_opencode_go_chat_completion(request).await;
        }
        let request = MessageRequest {
            stream: false,
            ..self.normalize_message_request(request)
        };
        let response = self.send_with_retry(&request).await?;
        let request_id = request_id_from_headers(response.headers());
        let mut response = response
            .json::<MessageResponse>()
            .await
            .map_err(ApiError::from)?;
        if response.request_id.is_none() {
            response.request_id = request_id;
        }
        Ok(response)
    }

    pub async fn send_chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, ApiError> {
        let request_url = format!(
            "{}{}",
            self.base_url.trim_end_matches('/'),
            self.chat_completions_path()
        );
        if nanogpt_client_debug_enabled() {
            let resolved_base_url = self.base_url.trim_end_matches('/');
            eprintln!("[nanogpt-client] resolved_base_url={resolved_base_url}");
            eprintln!("[nanogpt-client] request_url={request_url}");
        }
        let request_builder = self
            .http
            .post(&request_url)
            .header("content-type", "application/json");
        let request_builder = self.apply_auth_headers(request_builder, true);
        let response = request_builder
            .json(request)
            .send()
            .await
            .map_err(ApiError::from)?;
        let response = expect_success(response).await?;
        response
            .json::<ChatCompletionResponse>()
            .await
            .map_err(ApiError::from)
    }

    pub async fn stream_message(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageStream, ApiError> {
        if self.service == ApiService::OpencodeGo {
            let response = self.send_message(request).await?;
            return Ok(MessageStream::from_message_response(response));
        }
        let response = self
            .send_with_retry(&self.normalize_message_request(&request.clone().with_streaming()))
            .await?;
        Ok(MessageStream::from_http_response(response))
    }

    pub async fn fetch_models(&self, detailed: bool) -> Result<ModelsResponse, ApiError> {
        let response = self
            .send_get_request(
                "/v1/models",
                &[("detailed", if detailed { "true" } else { "false" })],
            )
            .await?;
        response
            .json::<ModelsResponse>()
            .await
            .map_err(ApiError::from)
    }

    pub async fn fetch_providers(
        &self,
        canonical_id: &str,
    ) -> Result<ProviderSelectionResponse, ApiError> {
        let request_url = providers_url(&self.base_url, canonical_id)?;
        if nanogpt_client_debug_enabled() {
            let resolved_base_url = self.base_url.trim_end_matches('/');
            eprintln!("[nanogpt-client] resolved_base_url={resolved_base_url}");
            eprintln!("[nanogpt-client] request_url={request_url}");
        }

        let request_builder = self.http.get(request_url);
        let request_builder = self.apply_auth_headers(request_builder, false);

        let response = request_builder.send().await.map_err(ApiError::from)?;
        let response = expect_success(response).await?;
        response
            .json::<ProviderSelectionResponse>()
            .await
            .map_err(ApiError::from)
    }

    async fn send_with_retry(
        &self,
        request: &MessageRequest,
    ) -> Result<reqwest::Response, ApiError> {
        let mut attempts = 0;
        let mut last_error: Option<ApiError>;

        loop {
            attempts += 1;
            match self.send_raw_request(request).await {
                Ok(response) => match expect_success(response).await {
                    Ok(response) => return Ok(response),
                    Err(error) if error.is_retryable() && attempts <= self.max_retries + 1 => {
                        last_error = Some(error);
                    }
                    Err(error) => return Err(error),
                },
                Err(error) if error.is_retryable() && attempts <= self.max_retries + 1 => {
                    last_error = Some(error);
                }
                Err(error) => return Err(error),
            }

            if attempts > self.max_retries {
                break;
            }

            tokio::time::sleep(self.backoff_for_attempt(attempts)?).await;
        }

        Err(ApiError::RetriesExhausted {
            attempts,
            last_error: Box::new(last_error.expect("retry loop must capture an error")),
        })
    }

    async fn send_raw_request(
        &self,
        request: &MessageRequest,
    ) -> Result<reqwest::Response, ApiError> {
        let request_url = format!(
            "{}{}",
            self.base_url.trim_end_matches('/'),
            self.messages_path()
        );
        if nanogpt_client_debug_enabled() {
            let resolved_base_url = self.base_url.trim_end_matches('/');
            eprintln!("[nanogpt-client] resolved_base_url={resolved_base_url}");
            eprintln!("[nanogpt-client] request_url={request_url}");
        }
        let request_builder = self
            .http
            .post(&request_url)
            .header("content-type", "application/json");
        let request_builder = self.apply_auth_headers(request_builder, true);

        request_builder
            .json(request)
            .send()
            .await
            .map_err(ApiError::from)
    }

    async fn send_chat_completion_raw(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<reqwest::Response, ApiError> {
        let request_url = format!(
            "{}{}",
            self.base_url.trim_end_matches('/'),
            self.chat_completions_path()
        );
        if nanogpt_client_debug_enabled() {
            let resolved_base_url = self.base_url.trim_end_matches('/');
            eprintln!("[nanogpt-client] resolved_base_url={resolved_base_url}");
            eprintln!("[nanogpt-client] request_url={request_url}");
        }
        let request_builder = self
            .http
            .post(&request_url)
            .header("content-type", "application/json");
        let request_builder = self.apply_auth_headers(request_builder, true);

        request_builder
            .json(request)
            .send()
            .await
            .map_err(ApiError::from)
    }

    async fn send_get_request(
        &self,
        path: &str,
        query: &[(&str, &str)],
    ) -> Result<reqwest::Response, ApiError> {
        let request_url = format!("{}{}", self.base_url.trim_end_matches('/'), path);
        if nanogpt_client_debug_enabled() {
            let resolved_base_url = self.base_url.trim_end_matches('/');
            eprintln!("[nanogpt-client] resolved_base_url={resolved_base_url}");
            eprintln!("[nanogpt-client] request_url={request_url}");
        }

        let request_builder = self.http.get(&request_url).query(query);
        let request_builder = self.apply_auth_headers(request_builder, false);

        let response = request_builder.send().await.map_err(ApiError::from)?;
        expect_success(response).await
    }

    fn apply_auth_headers(
        &self,
        request_builder: reqwest::RequestBuilder,
        include_provider: bool,
    ) -> reqwest::RequestBuilder {
        let debug = nanogpt_client_debug_enabled();
        let request_builder = if self.api_key.is_empty() {
            if debug {
                eprintln!("[nanogpt-client] headers authorization=<absent> x-api-key=<absent>");
            }
            request_builder
        } else {
            if debug {
                match self.service {
                    ApiService::NanoGpt => eprintln!(
                        "[nanogpt-client] headers x-api-key=[REDACTED] authorization=Bearer [REDACTED]"
                    ),
                    ApiService::Synthetic => eprintln!(
                        "[nanogpt-client] headers authorization=Bearer [REDACTED]"
                    ),
                    ApiService::OpencodeGo => eprintln!(
                        "[nanogpt-client] headers x-api-key=[REDACTED] authorization=Bearer [REDACTED]"
                    ),
                }
            }
            match self.service {
                ApiService::NanoGpt => request_builder
                    .bearer_auth(&self.api_key)
                    .header("x-api-key", &self.api_key),
                ApiService::Synthetic => request_builder.bearer_auth(&self.api_key),
                ApiService::OpencodeGo => request_builder
                    .bearer_auth(&self.api_key)
                    .header("x-api-key", &self.api_key),
            }
        };

        if include_provider {
            if let Some(provider) = &self.provider {
                if debug {
                    eprintln!("[nanogpt-client] x-provider={provider}");
                }
                let request_builder = request_builder.header("x-provider", provider);
                if self.force_paygo {
                    if debug {
                        eprintln!("[nanogpt-client] x-billing-mode=paygo");
                    }
                    return request_builder.header("x-billing-mode", "paygo");
                }
                return request_builder;
            }
        }
        request_builder
    }

    fn backoff_for_attempt(&self, attempt: u32) -> Result<Duration, ApiError> {
        let Some(multiplier) = 1_u32.checked_shl(attempt.saturating_sub(1)) else {
            return Err(ApiError::BackoffOverflow {
                attempt,
                base_delay: self.initial_backoff,
            });
        };
        Ok(self
            .initial_backoff
            .checked_mul(multiplier)
            .map_or(self.max_backoff, |delay| delay.min(self.max_backoff)))
    }

    fn messages_path(&self) -> &'static str {
        match self.service {
            ApiService::NanoGpt => "/v1/messages",
            ApiService::Synthetic => "/messages",
            ApiService::OpencodeGo => "/v1/messages",
        }
    }

    fn chat_completions_path(&self) -> &'static str {
        match self.service {
            ApiService::NanoGpt => "/v1/chat/completions",
            ApiService::Synthetic => "/chat/completions",
            ApiService::OpencodeGo => "/v1/chat/completions",
        }
    }

    fn normalize_message_request(&self, request: &MessageRequest) -> MessageRequest {
        let mut normalized = request.clone();
        normalized.model = self.normalize_model_id(&normalized.model);
        normalized
    }

    fn normalize_model_id(&self, model: &str) -> String {
        match self.service {
            ApiService::OpencodeGo => normalize_opencode_go_model_id(model).to_string(),
            ApiService::NanoGpt | ApiService::Synthetic => model.to_string(),
        }
    }

    async fn send_opencode_go_chat_completion(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageResponse, ApiError> {
        let chat_request = message_request_to_chat_completion_request(request, self)?;
        let response = self.send_chat_completion_with_retry(&chat_request).await?;
        let request_id = request_id_from_headers(response.headers());
        let response = response
            .json::<ChatCompletionResponse>()
            .await
            .map_err(ApiError::from)?;
        Ok(chat_completion_to_message_response(
            response,
            request.model.clone(),
            request_id,
        ))
    }

    async fn send_chat_completion_with_retry(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<reqwest::Response, ApiError> {
        let mut attempts = 0;
        let mut last_error: Option<ApiError>;

        loop {
            attempts += 1;
            match self.send_chat_completion_raw(request).await {
                Ok(response) => match expect_success(response).await {
                    Ok(response) => return Ok(response),
                    Err(error) if error.is_retryable() && attempts <= self.max_retries + 1 => {
                        last_error = Some(error);
                    }
                    Err(error) => return Err(error),
                },
                Err(error) if error.is_retryable() && attempts <= self.max_retries + 1 => {
                    last_error = Some(error);
                }
                Err(error) => return Err(error),
            }

            if attempts > self.max_retries {
                break;
            }

            tokio::time::sleep(self.backoff_for_attempt(attempts)?).await;
        }

        Err(ApiError::RetriesExhausted {
            attempts,
            last_error: Box::new(last_error.expect("retry loop must capture an error")),
        })
    }
}

fn read_api_key() -> Result<String, ApiError> {
    resolve_api_key_for(ApiService::NanoGpt)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiService {
    NanoGpt,
    Synthetic,
    OpencodeGo,
}

impl ApiService {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NanoGpt => "nanogpt",
            Self::Synthetic => "synthetic",
            Self::OpencodeGo => "opencode_go",
        }
    }

    #[must_use]
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::NanoGpt => "NanoGPT",
            Self::Synthetic => "Synthetic",
            Self::OpencodeGo => "OpenCode Go",
        }
    }
}

pub fn resolve_api_key_for(service: ApiService) -> Result<String, ApiError> {
    match std::env::var(service_api_key_env(service)) {
        Ok(api_key) if !api_key.is_empty() => Ok(api_key),
        Ok(_) => Err(ApiError::MissingApiKey),
        Err(std::env::VarError::NotPresent) => {
            read_api_key_from_credentials_file(service).ok_or(ApiError::MissingApiKey)
        }
        Err(error) => Err(ApiError::from(error)),
    }
}

pub fn resolve_api_key() -> Result<String, ApiError> {
    read_api_key()
}

pub fn resolve_base_url_for(service: ApiService) -> String {
    match service {
        ApiService::NanoGpt => {
            std::env::var("NANOGPT_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string())
        }
        ApiService::Synthetic => std::env::var("SYNTHETIC_BASE_URL")
            .unwrap_or_else(|_| DEFAULT_SYNTHETIC_MESSAGES_BASE_URL.to_string()),
        ApiService::OpencodeGo => std::env::var("OPENCODE_GO_BASE_URL")
            .unwrap_or_else(|_| DEFAULT_OPENCODE_GO_BASE_URL.to_string()),
    }
}

pub fn resolve_root_url_for(service: ApiService) -> String {
    match service {
        ApiService::NanoGpt => {
            let base = resolve_base_url_for(service);
            let trimmed = base.trim_end_matches('/');
            trimmed.strip_suffix("/api").unwrap_or(trimmed).to_string()
        }
        ApiService::Synthetic => {
            if let Ok(root) = std::env::var("SYNTHETIC_ROOT_URL") {
                return root;
            }
            let base = resolve_base_url_for(service);
            let trimmed = base.trim_end_matches('/');
            trimmed
                .strip_suffix("/anthropic/v1")
                .unwrap_or(trimmed)
                .to_string()
        }
        ApiService::OpencodeGo => resolve_base_url_for(service),
    }
}

fn read_api_key_from_credentials_file(service: ApiService) -> Option<String> {
    let path = credentials_path()?;
    let contents = fs::read_to_string(path).ok()?;
    let parsed = serde_json::from_str::<serde_json::Value>(&contents).ok()?;
    let service_key = match service {
        ApiService::NanoGpt => "nanogpt_api_key",
        ApiService::Synthetic => "synthetic_api_key",
        ApiService::OpencodeGo => "opencode_go_api_key",
    };
    parsed
        .get(service_key)
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            (service == ApiService::NanoGpt)
                .then(|| parsed.get("apiKey").and_then(serde_json::Value::as_str))
                .flatten()
        })
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn service_api_key_env(service: ApiService) -> &'static str {
    match service {
        ApiService::NanoGpt => "NANOGPT_API_KEY",
        ApiService::Synthetic => "SYNTHETIC_API_KEY",
        ApiService::OpencodeGo => "OPENCODE_GO_API_KEY",
    }
}

fn credentials_path() -> Option<PathBuf> {
    Some(pebble_config_home()?.join("credentials.json"))
}

fn request_id_from_headers(headers: &reqwest::header::HeaderMap) -> Option<String> {
    headers
        .get(REQUEST_ID_HEADER)
        .or_else(|| headers.get(ALT_REQUEST_ID_HEADER))
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned)
}

fn providers_url(base_url: &str, canonical_id: &str) -> Result<reqwest::Url, ApiError> {
    let mut url =
        reqwest::Url::parse(&format!("{}/", base_url.trim_end_matches('/'))).map_err(|error| {
            ApiError::Io(std::io::Error::new(std::io::ErrorKind::InvalidInput, error))
        })?;
    let mut segments = url.path_segments_mut().map_err(|_| {
        ApiError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "invalid base url",
        ))
    })?;
    segments.pop_if_empty();
    segments.push("models");
    segments.push(canonical_id);
    segments.push("providers");
    drop(segments);
    Ok(url)
}

#[derive(Debug)]
pub struct MessageStream {
    request_id: Option<String>,
    state: MessageStreamState,
    pending: VecDeque<StreamEvent>,
}

#[derive(Debug)]
enum MessageStreamState {
    Http {
        response: reqwest::Response,
        parser: SseParser,
        done: bool,
    },
    Buffered,
}

impl MessageStream {
    fn from_http_response(response: reqwest::Response) -> Self {
        Self {
            request_id: request_id_from_headers(response.headers()),
            state: MessageStreamState::Http {
                response,
                parser: SseParser::new(),
                done: false,
            },
            pending: VecDeque::new(),
        }
    }

    fn from_message_response(response: MessageResponse) -> Self {
        Self {
            request_id: response.request_id.clone(),
            state: MessageStreamState::Buffered,
            pending: VecDeque::from(message_response_to_stream_events(response)),
        }
    }

    #[must_use]
    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    pub async fn next_event(&mut self) -> Result<Option<StreamEvent>, ApiError> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Ok(Some(event));
            }

            match &mut self.state {
                MessageStreamState::Buffered => return Ok(None),
                MessageStreamState::Http {
                    response,
                    parser,
                    done,
                } => {
                    if *done {
                        let remaining = parser.finish()?;
                        self.pending.extend(remaining);
                        if let Some(event) = self.pending.pop_front() {
                            return Ok(Some(event));
                        }
                        return Ok(None);
                    }

                    match response.chunk().await? {
                        Some(chunk) => {
                            self.pending.extend(parser.push(&chunk)?);
                        }
                        None => {
                            *done = true;
                        }
                    }
                }
            }
        }
    }
}

fn normalize_opencode_go_model_id(model: &str) -> &str {
    model.strip_prefix("opencode-go/").unwrap_or(model)
}

fn opencode_go_uses_messages_api(model: &str) -> bool {
    matches!(
        normalize_opencode_go_model_id(model),
        "minimax-m2.5" | "minimax-m2.7"
    )
}

fn opencode_go_prefers_thinking_disabled(model: &str) -> bool {
    matches!(
        normalize_opencode_go_model_id(model),
        "kimi-k2.5" | "kimi-k2.6"
    )
}

fn invalid_request_error(message: impl Into<String>) -> ApiError {
    ApiError::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        message.into(),
    ))
}

fn message_request_to_chat_completion_request(
    request: &MessageRequest,
    client: &NanoGptClient,
) -> Result<ChatCompletionRequest, ApiError> {
    let mut messages = Vec::new();
    if let Some(system) = request.system.as_ref().filter(|system| !system.is_empty()) {
        messages.push(ChatCompletionMessage {
            role: "system".to_string(),
            content: Some(ChatCompletionContent::Text(system.clone())),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            reasoning: None,
        });
    }

    for message in &request.messages {
        messages.extend(input_message_to_chat_completion_messages(message)?);
    }

    Ok(ChatCompletionRequest {
        model: client.normalize_model_id(&request.model),
        messages,
        max_tokens: Some(request.max_tokens),
        tools: request.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|tool| ChatCompletionTool {
                    kind: "function".to_string(),
                    function: ChatCompletionFunction {
                        name: tool.name.clone(),
                        description: tool.description.clone(),
                        parameters: Some(tool.input_schema.clone()),
                    },
                })
                .collect()
        }),
        tool_choice: request.tool_choice.as_ref().map(map_tool_choice),
        billing_mode: None,
        thinking: (client.service == ApiService::OpencodeGo
            && opencode_go_prefers_thinking_disabled(&request.model))
        .then(ChatCompletionThinkingConfig::disabled),
        stream: false,
    })
}

fn input_message_to_chat_completion_messages(
    message: &crate::types::InputMessage,
) -> Result<Vec<ChatCompletionMessage>, ApiError> {
    match message.role.as_str() {
        "assistant" => assistant_input_to_chat_completion_messages(message),
        "user" => user_input_to_chat_completion_messages(message),
        other => Err(invalid_request_error(format!(
            "unsupported role for chat/completions translation: {other}"
        ))),
    }
}

fn assistant_input_to_chat_completion_messages(
    message: &crate::types::InputMessage,
) -> Result<Vec<ChatCompletionMessage>, ApiError> {
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for block in &message.content {
        match block {
            crate::types::InputContentBlock::Text { text } => {
                if !text.is_empty() {
                    text_parts.push(text.clone());
                }
            }
            crate::types::InputContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(crate::types::ChatCompletionToolCall {
                    id: id.clone(),
                    kind: "function".to_string(),
                    function: crate::types::ChatCompletionFunctionCall {
                        name: name.clone(),
                        arguments: input.to_string(),
                    },
                });
            }
            crate::types::InputContentBlock::Image { .. } => {
                return Err(invalid_request_error(
                    "image inputs are not supported for OpenCode Go chat/completions models",
                ));
            }
            crate::types::InputContentBlock::ToolResult { .. } => {
                return Err(invalid_request_error(
                    "assistant tool_result blocks cannot be translated to chat/completions",
                ));
            }
        }
    }

    Ok(vec![ChatCompletionMessage {
        role: "assistant".to_string(),
        content: (!text_parts.is_empty())
            .then(|| ChatCompletionContent::Text(text_parts.join("\n\n"))),
        tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
        tool_call_id: None,
        reasoning_content: message.reasoning_content.clone(),
        reasoning: message
            .reasoning
            .clone()
            .or(message.reasoning_content.clone()),
    }])
}

fn user_input_to_chat_completion_messages(
    message: &crate::types::InputMessage,
) -> Result<Vec<ChatCompletionMessage>, ApiError> {
    let mut messages = Vec::new();
    let mut pending_text = Vec::new();

    for block in &message.content {
        match block {
            crate::types::InputContentBlock::Text { text } => {
                if !text.is_empty() {
                    pending_text.push(text.clone());
                }
            }
            crate::types::InputContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                if !pending_text.is_empty() {
                    messages.push(ChatCompletionMessage {
                        role: "user".to_string(),
                        content: Some(ChatCompletionContent::Text(pending_text.join("\n\n"))),
                        tool_calls: None,
                        tool_call_id: None,
                        reasoning_content: None,
                        reasoning: None,
                    });
                    pending_text.clear();
                }
                messages.push(ChatCompletionMessage {
                    role: "tool".to_string(),
                    content: Some(ChatCompletionContent::Text(tool_result_content_to_string(
                        content,
                    )?)),
                    tool_calls: None,
                    tool_call_id: Some(tool_use_id.clone()),
                    reasoning_content: None,
                    reasoning: None,
                });
            }
            crate::types::InputContentBlock::Image { .. } => {
                return Err(invalid_request_error(
                    "image inputs are not supported for OpenCode Go chat/completions models",
                ));
            }
            crate::types::InputContentBlock::ToolUse { .. } => {
                return Err(invalid_request_error(
                    "user tool_use blocks cannot be translated to chat/completions",
                ));
            }
        }
    }

    if !pending_text.is_empty() {
        messages.push(ChatCompletionMessage {
            role: "user".to_string(),
            content: Some(ChatCompletionContent::Text(pending_text.join("\n\n"))),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            reasoning: None,
        });
    }

    Ok(messages)
}

fn tool_result_content_to_string(
    content: &[crate::types::ToolResultContentBlock],
) -> Result<String, ApiError> {
    let mut parts = Vec::new();
    for block in content {
        match block {
            crate::types::ToolResultContentBlock::Text { text } => parts.push(text.clone()),
            crate::types::ToolResultContentBlock::Json { value } => parts.push(value.to_string()),
        }
    }
    if parts.is_empty() {
        return Err(invalid_request_error(
            "tool result content cannot be empty for chat/completions translation",
        ));
    }
    Ok(parts.join("\n"))
}

fn map_tool_choice(choice: &crate::types::ToolChoice) -> crate::types::ChatCompletionToolChoice {
    match choice {
        crate::types::ToolChoice::Auto => {
            crate::types::ChatCompletionToolChoice::Mode("auto".to_string())
        }
        crate::types::ToolChoice::Any => {
            crate::types::ChatCompletionToolChoice::Mode("required".to_string())
        }
        crate::types::ToolChoice::Tool { name } => {
            crate::types::ChatCompletionToolChoice::Function {
                kind: "function".to_string(),
                function: crate::types::ChatCompletionNamedFunction { name: name.clone() },
            }
        }
    }
}

fn chat_completion_to_message_response(
    response: ChatCompletionResponse,
    requested_model: String,
    request_id: Option<String>,
) -> MessageResponse {
    let choice =
        response
            .choices
            .into_iter()
            .next()
            .unwrap_or(crate::types::ChatCompletionChoice {
                index: 0,
                message: crate::types::ChatCompletionAssistantMessage {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: None,
                    reasoning_content: None,
                    reasoning: None,
                },
                finish_reason: None,
            });

    let mut content = Vec::new();
    if let Some(reasoning) = choice
        .message
        .reasoning
        .clone()
        .or(choice.message.reasoning_content.clone())
        .filter(|text| !text.is_empty())
    {
        content.push(OutputContentBlock::Thinking {
            thinking: reasoning,
            signature: None,
        });
    }
    if let Some(text) = choice
        .message
        .content
        .as_ref()
        .and_then(chat_completion_content_to_text)
        .filter(|text| !text.is_empty())
    {
        content.push(OutputContentBlock::Text { text });
    }
    if let Some(tool_calls) = choice.message.tool_calls {
        for tool_call in tool_calls {
            let input = serde_json::from_str(&tool_call.function.arguments)
                .unwrap_or_else(|_| serde_json::Value::String(tool_call.function.arguments));
            content.push(OutputContentBlock::ToolUse {
                id: tool_call.id,
                name: tool_call.function.name,
                input,
            });
        }
    }

    let usage = response.usage.unwrap_or(crate::types::ChatCompletionUsage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    });

    MessageResponse {
        id: response.id,
        kind: "message".to_string(),
        role: choice.message.role,
        content,
        model: requested_model,
        stop_reason: choice.finish_reason,
        stop_sequence: None,
        usage: crate::types::Usage {
            input_tokens: usage.prompt_tokens,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
            output_tokens: usage.completion_tokens,
        },
        request_id,
    }
}

fn chat_completion_content_to_text(content: &ChatCompletionContent) -> Option<String> {
    match content {
        ChatCompletionContent::Text(text) => Some(text.clone()),
        ChatCompletionContent::Parts(parts) => {
            let text = parts
                .iter()
                .filter(|part| part.kind == "text" || part.kind == "output_text")
                .filter_map(|part| part.text.as_deref())
                .collect::<Vec<_>>()
                .join("\n");
            (!text.is_empty()).then_some(text)
        }
    }
}

fn message_response_to_stream_events(response: MessageResponse) -> Vec<StreamEvent> {
    let usage = response.usage.clone();
    let delta = crate::types::MessageDelta {
        stop_reason: response.stop_reason.clone(),
        stop_sequence: response.stop_sequence.clone(),
    };
    let content_blocks = response.content.clone();
    let message = MessageResponse {
        content: Vec::new(),
        ..response
    };
    let mut events = vec![StreamEvent::MessageStart(crate::types::MessageStartEvent {
        message,
    })];

    for (index, block) in content_blocks.into_iter().enumerate() {
        let index = index as u32;
        events.push(StreamEvent::ContentBlockStart(
            crate::types::ContentBlockStartEvent {
                index,
                content_block: block,
            },
        ));
        events.push(StreamEvent::ContentBlockStop(
            crate::types::ContentBlockStopEvent { index },
        ));
    }

    events.push(StreamEvent::MessageDelta(crate::types::MessageDeltaEvent {
        delta,
        usage,
    }));
    events.push(StreamEvent::MessageStop(crate::types::MessageStopEvent {}));
    events
}

async fn expect_success(response: reqwest::Response) -> Result<reqwest::Response, ApiError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }

    let body = response.text().await.unwrap_or_else(|_| String::new());
    let parsed_error = serde_json::from_str::<NanoGptErrorEnvelope>(&body).ok();
    let retryable = is_retryable_status(status);

    Err(ApiError::Api {
        status,
        error_type: parsed_error
            .as_ref()
            .map(|error| error.error.error_type.clone()),
        message: parsed_error
            .as_ref()
            .map(|error| error.error.message.clone()),
        body,
        retryable,
    })
}

const fn is_retryable_status(status: reqwest::StatusCode) -> bool {
    matches!(status.as_u16(), 408 | 409 | 429 | 500 | 502 | 503 | 504)
}

#[derive(Debug, Deserialize)]
struct NanoGptErrorEnvelope {
    error: NanoGptErrorBody,
}

#[derive(Debug, Deserialize)]
struct NanoGptErrorBody {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

#[cfg(test)]
mod tests {
    use super::{ALT_REQUEST_ID_HEADER, REQUEST_ID_HEADER};
    use std::sync::{Mutex, OnceLock};
    use std::time::Duration;

    use crate::types::{ContentBlockDelta, MessageRequest};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env test lock should not be poisoned")
    }

    fn temp_config_home() -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "pebble-api-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time should be after epoch")
                .as_nanos()
        ))
    }

    #[test]
    fn read_api_key_requires_presence() {
        let _guard = env_lock();
        let root = temp_config_home();
        std::fs::create_dir_all(&root).expect("config dir should exist");
        std::env::remove_var("NANOGPT_API_KEY");
        std::env::set_var("PEBBLE_CONFIG_HOME", &root);
        let error = super::read_api_key().expect_err("missing key should error");
        assert!(matches!(error, crate::error::ApiError::MissingApiKey));
        std::env::remove_var("PEBBLE_CONFIG_HOME");
        std::fs::remove_dir_all(root).expect("temp config dir should be removed");
    }

    #[test]
    fn read_api_key_requires_non_empty_value() {
        let _guard = env_lock();
        let root = temp_config_home();
        std::fs::create_dir_all(&root).expect("config dir should exist");
        std::env::set_var("NANOGPT_API_KEY", "");
        std::env::set_var("PEBBLE_CONFIG_HOME", &root);
        let error = super::read_api_key().expect_err("empty key should error");
        assert!(matches!(error, crate::error::ApiError::MissingApiKey));
        std::env::remove_var("NANOGPT_API_KEY");
        std::env::remove_var("PEBBLE_CONFIG_HOME");
        std::fs::remove_dir_all(root).expect("temp config dir should be removed");
    }

    #[test]
    fn read_api_key_uses_nanogpt_env() {
        let _guard = env_lock();
        let root = temp_config_home();
        std::fs::create_dir_all(&root).expect("config dir should exist");
        std::env::set_var("NANOGPT_API_KEY", "nano-key");
        std::env::set_var("PEBBLE_CONFIG_HOME", &root);
        assert_eq!(
            super::read_api_key().expect("api key should load"),
            "nano-key"
        );
        std::env::remove_var("NANOGPT_API_KEY");
        std::env::remove_var("PEBBLE_CONFIG_HOME");
        std::fs::remove_dir_all(root).expect("temp config dir should be removed");
    }

    #[test]
    fn read_base_url_defaults_to_nanogpt_messages_api_root() {
        let _guard = env_lock();
        std::env::remove_var("NANOGPT_BASE_URL");
        assert_eq!(
            super::resolve_base_url_for(super::ApiService::NanoGpt),
            "https://nano-gpt.com/api"
        );
    }

    #[test]
    fn read_api_key_uses_pebble_credentials_file() {
        let _guard = env_lock();
        let root = temp_config_home();
        std::fs::create_dir_all(&root).expect("config dir should exist");
        std::fs::write(
            root.join("credentials.json"),
            r#"{"nanogpt_api_key":"from-credentials"}"#,
        )
        .expect("credentials should write");

        std::env::remove_var("NANOGPT_API_KEY");
        std::env::set_var("PEBBLE_CONFIG_HOME", &root);
        assert_eq!(
            super::read_api_key().expect("api key should load"),
            "from-credentials"
        );

        std::env::remove_var("PEBBLE_CONFIG_HOME");
        std::fs::remove_dir_all(root).expect("temp config dir should be removed");
    }

    #[test]
    fn message_request_stream_helper_sets_stream_true() {
        let request = MessageRequest {
            model: "openai/gpt-5.2".to_string(),
            max_tokens: 64,
            messages: vec![],
            system: None,
            tools: None,
            tool_choice: None,
            thinking: None,
            stream: false,
        };

        assert!(request.with_streaming().stream);
    }

    #[test]
    fn backoff_doubles_until_maximum() {
        let client = super::NanoGptClient::new("test-key").with_retry_policy(
            3,
            Duration::from_millis(10),
            Duration::from_millis(25),
        );
        assert_eq!(
            client.backoff_for_attempt(1).expect("attempt 1"),
            Duration::from_millis(10)
        );
        assert_eq!(
            client.backoff_for_attempt(2).expect("attempt 2"),
            Duration::from_millis(20)
        );
        assert_eq!(
            client.backoff_for_attempt(3).expect("attempt 3"),
            Duration::from_millis(25)
        );
    }

    #[test]
    fn retryable_statuses_are_detected() {
        assert!(super::is_retryable_status(
            reqwest::StatusCode::TOO_MANY_REQUESTS
        ));
        assert!(super::is_retryable_status(
            reqwest::StatusCode::INTERNAL_SERVER_ERROR
        ));
        assert!(!super::is_retryable_status(
            reqwest::StatusCode::UNAUTHORIZED
        ));
    }

    #[test]
    fn tool_delta_variant_round_trips() {
        let delta = ContentBlockDelta::InputJsonDelta {
            partial_json: "{\"city\":\"Paris\"}".to_string(),
        };
        let encoded = serde_json::to_string(&delta).expect("delta should serialize");
        let decoded: ContentBlockDelta =
            serde_json::from_str(&encoded).expect("delta should deserialize");
        assert_eq!(decoded, delta);
    }

    #[test]
    fn request_id_uses_primary_or_fallback_header() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(REQUEST_ID_HEADER, "req_primary".parse().expect("header"));
        assert_eq!(
            super::request_id_from_headers(&headers).as_deref(),
            Some("req_primary")
        );

        headers.clear();
        headers.insert(
            ALT_REQUEST_ID_HEADER,
            "req_fallback".parse().expect("header"),
        );
        assert_eq!(
            super::request_id_from_headers(&headers).as_deref(),
            Some("req_fallback")
        );
    }
}

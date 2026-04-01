mod client;
mod error;
mod sse;
mod types;

pub use client::{resolve_api_key, MessageStream, NanoGptClient};
pub use error::ApiError;
pub use sse::{parse_frame, SseParser};
pub use types::{
    ChatCompletionAssistantMessage, ChatCompletionChoice, ChatCompletionFunction,
    ChatCompletionFunctionCall, ChatCompletionMessage, ChatCompletionNamedFunction,
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionTool, ChatCompletionToolCall,
    ChatCompletionToolChoice, ChatCompletionUsage, ContentBlockDelta, ContentBlockDeltaEvent,
    ContentBlockStartEvent, ContentBlockStopEvent, ImageSource, InputContentBlock, InputMessage,
    MessageDelta, MessageDeltaEvent, MessageRequest, MessageResponse, MessageStartEvent,
    MessageStopEvent, ModelCapabilities, ModelInfo, ModelPricing, ModelProvider, ModelsResponse,
    OutputContentBlock, ProviderPrice, ProviderSelectionResponse, StreamEvent, ToolChoice,
    ToolDefinition, ToolResultContentBlock, Usage,
};

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Callable

@dataclass
class MessageContentPart:
    type: str  # "text", "image", "file"
    text: Optional[str] = None
    image: Optional[Dict[str, Any]] = None  # {"url": str, "mimeType": Optional[str]}
    file: Optional[Dict[str, Any]] = None   # {"url": str, "mimeType": str}

@dataclass
class Message:
    role: str  # "system", "user", "assistant"
    content: Union[str, List[MessageContentPart]]

@dataclass
class TokenUsage:
    prompt: int = 0
    completion: int = 0
    total: int = 0

@dataclass
class ToolDefinition:
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

@dataclass
class ToolCall:
    id: str
    type: str  # "function"
    function: Dict[str, str]  # {"name": str, "arguments": str}

@dataclass
class ResponseFormat:
    type: str  # "text", "json_object", "json_schema"
    schema: Optional[Dict[str, Any]] = None

@dataclass
class ChatRequest:
    message: Optional[str] = None
    messages: Optional[List[Message]] = None
    system_prompt: Optional[str] = None
    provider: str = "auto"
    model: Optional[str] = None
    goal: Union[str, Callable[[List[Any], Any], Any]] = "auto"
    fallback: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False
    cache: bool = True
    tools: Optional[List[ToolDefinition]] = None
    response_format: Optional[Union[str, ResponseFormat]] = None

@dataclass
class ChatResponse:
    content: str
    provider: str
    model: str
    usage: TokenUsage
    latency: int
    cost: float = 0.0
    finish_reason: str = "stop"
    tool_calls: Optional[List[ToolCall]] = None
    json: Optional[Dict[str, Any]] = None

@dataclass
class StreamChunk:
    delta: str
    done: bool
    provider: str
    model: str

@dataclass
class ProviderConfig:
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: Optional[int] = None
    default_model: Optional[str] = None

@dataclass
class CircuitBreakerConfig:
    failures_threshold: int = 3
    cooldown_seconds: int = 60

@dataclass
class VyrionConfig:
    timeout: Optional[int] = None
    fallback: Optional[List[str]] = None
    default_goal: str = "auto"
    cache: Union[bool, Any] = False
    circuit_breaker: Optional[CircuitBreakerConfig] = None

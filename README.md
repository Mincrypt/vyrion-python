# Vyrion (Python SDK)

One intelligent runtime for every LLM. Unified async API, smart routing, auto failover, streaming caching, circuit breakers, and cost estimation in Python.

## Features

- **Unified Async API**: One async interface for chat and streaming.
- **Smart Routing & Fallbacks**: Automatically select and retry alternate providers.
- **Circuit Breaker**: Auto-trip and temporary bypass for rate-limited (HTTP 429) or degraded engines.
- **Streaming Cache & Playback**: Cache streaming responses and play back chunks at a controlled `15ms` interval.
- **Multi-Modal Data**: Standardized support for text, base64 images, and documents.
- **Onion Middleware**: Intercept and modify requests/responses dynamically.

## Installation

Install via pip:

```bash
pip install vyrion

# Install only what you use (optional extras)
pip install vyrion[openai,gemini,anthropic,groq]
```

## Quick Start

```python
import asyncio
from vyrion import Vyrion

async def main():
    # Configure the client
    ai = Vyrion(
        openai="your-openai-api-key",
        gemini="your-gemini-api-key"
    )

    # Simple chat
    response = await ai.chat(
        message="Explain recursion in one sentence.",
        goal="fastest"
    )
    print(f"[{response.provider}]: {response.content}")

    # Async streaming
    async for chunk in ai.stream(message="Tell me a story"):
        print(chunk.delta, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Global Configuration (`VyrionConfig`)

Passed to `Vyrion(**config)`:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `openai` | `str \| dict` | `None` | OpenAI API Key or provider configuration dictionary. |
| `gemini` | `str \| dict` | `None` | Gemini API Key or provider configuration dictionary. |
| `anthropic` | `str \| dict` | `None` | Anthropic API Key or provider configuration dictionary. |
| `groq` | `str \| dict` | `None` | Groq API Key or provider configuration dictionary. |
| `ollama` | `dict` | `None` | Ollama local configuration dictionary. |
| `timeout` | `int` | `30` | Request timeout override in seconds. |
| `fallback` | `list[str]` | *(priority list)* | Fallback sequence order on provider failures. |
| `default_goal` | `str` | `"auto"` | Default goal strategy (`"auto" \| "fastest" \| "cheapest" \| "best"`). |
| `cache` | `bool \| object` | `False` | Enables memory cache or registers custom cache backend. |
| `circuit_breaker` | `dict` | `None` | Settings mapping `failures_threshold` and `cooldown_seconds`. |

### Provider overrides (e.g. `gemini={"default_model": "...", "timeout": 15}`)

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `api_key` | `str` | Specific API authentication key. |
| `base_url` | `str` | Overrides base API url (useful for local models/proxies). |
| `timeout` | `int` | Override request timeout in seconds. |
| `default_model` | `str` | Override default model name. |

---

## Request Configuration (`ChatRequest`)

Passed to `ai.chat(request)` and `ai.stream(request)` as a dictionary or `ChatRequest` model:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `message` | `str` | `None` | User prompt string (single-turn shorthand). |
| `messages` | `list[Message]` | `[]` | Multi-turn chat messages. Supports text, image, and file structures. |
| `system_prompt` | `str` | `None` | System instruction prepended to conversation. |
| `provider` | `str` | `"auto"` | Target provider name or `"auto"`. |
| `model` | `str` | *(default)* | Override model name to use. |
| `goal` | `str \| Callable` | `"auto"` | Routing strategy goal or custom strategy function. |
| `fallback` | `list[str]` | *(global fallback)* | Per-request fallback list override. |
| `max_tokens` | `int` | `None` | Maximum token length of generated response. |
| `temperature` | `float` | `None` | Sampling temperature between 0 and 2. |
| `cache` | `bool` | `True` | Set to `False` to force bypass cache checks. |
| `tools` | `list[dict]` | `None` | List of functional tool declarations. |
| `response_format` | `str \| dict` | `None` | Enforces JSON object / schema formatting. |

---

## Multi-Modal & File Attachments

Send images, PDFs, Word documents, or text files in a standardized parts array:

```python
response = await ai.chat(
    messages=[
        {
            "role": "user",
            "content": [
                { "type": "text", "text": "Analyze this file and image:" },
                {
                    "type": "file",
                    "file": {
                        "url": "data:application/pdf;base64,JVBERi...", # PDF base64
                        "mimeType": "application/pdf"
                    }
                },
                {
                    "type": "image",
                    "image": {
                        "url": "data:image/jpeg;base64,...", # image base64
                        "mimeType": "image/jpeg"
                    }
                }
            ]
        }
    ]
)
```

---

## Circuit Breaker Configuration

```python
ai = Vyrion(
    openai="sk-...",
    groq="gsk_...",
    circuit_breaker={
        "failures_threshold": 2,      # Trip after 2 consecutive errors
        "cooldown_seconds": 30        # Bypass for 30 seconds
    }
)
```

---

## Caching & Simulated Playback

Caching caches both `chat()` and `stream()` responses. On a cache hit for a stream, chunks are played back with a `15ms` simulated delay:

```python
ai = Vyrion(openai="sk-...", cache=True)

# Miss: fetches and caches
async for chunk in ai.stream(message="Hello"):
    print(chunk.delta)

# Hit: plays back from cache smoothly
async for chunk in ai.stream(message="Hello"):
    print(chunk.delta)
```

---

## Onion Middleware

Interceptors wrap request/response cycles:

```python
# Create a logging middleware
async def logging_middleware(ctx, next_dispatch):
    print(f"Sending request to provider: {ctx['request'].provider}")
    start = time.time()
    response = await next_dispatch()
    print(f"Finished in {int((time.time() - start) * 1000)}ms")
    return response

ai.use(logging_middleware)
```

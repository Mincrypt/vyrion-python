import base64
import time
from typing import Union, List, Any, AsyncIterator
from .base import BaseProvider
from ..types import ChatRequest, ChatResponse, StreamChunk, Message, MessageContentPart, TokenUsage

class MistralProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "mistral"

    @property
    def default_model(self) -> str:
        return "mistral-small-latest"

    @property
    def supported_models(self) -> List[str]:
        return [
            "mistral-large-latest",
            "mistral-medium-latest",
            "mistral-small-latest",
            "mistral-tiny",
            "codestral-latest",
            "mistral-embed",
            "open-mistral-nemo",
            "open-mixtral-8x22b",
            "open-mixtral-8x7b",
        ]

    def __init__(self, config: dict):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if not self._client:
            try:
                from mistralai import Mistral
            except ImportError:
                raise ImportError(
                    "The 'mistralai' package is required for MistralProvider. "
                    "Install it using 'pip install vyrion[mistral]'"
                )
            self._client = Mistral(
                api_key=self.config.get("api_key"),
                server_url=self.config.get("base_url"),
            )
        return self._client

    async def chat(self, req: ChatRequest) -> ChatResponse:
        client = self._get_client()
        model = self.resolve_model(req)
        
        messages = []
        for m in self.build_messages(req):
            messages.append({
                "role": m.role,
                "content": self._map_content(m.content)
            })

        tools = None
        if req.tools:
            tools = [{
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
            } for t in req.tools]

        response_format = None
        if req.response_format:
            if req.response_format == "json" or getattr(req.response_format, "type", None) == "json_object":
                response_format = {"type": "json_object"}
            elif getattr(req.response_format, "type", None) == "json_schema":
                response_format = {"type": "json_object"} # Mistral uses standard json_object for general json output

        response = await client.chat.complete_async(
            model=model,
            messages=messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            tools=tools,
            response_format=response_format,
        )

        choice = response.choices[0]
        content = choice.message.content or ""
        
        usage = TokenUsage(
            prompt=response.usage.prompt_tokens if response.usage else 0,
            completion=response.usage.completion_tokens if response.usage else 0,
            total=response.usage.total_tokens if response.usage else 0,
        )

        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [{
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            } for tc in choice.message.tool_calls]

        import json
        json_parsed = None
        if req.response_format and content:
            try:
                json_parsed = json.loads(content)
            except Exception:
                pass

        return ChatResponse(
            content=content,
            provider=self.name,
            model=model,
            usage=usage,
            latency=0,
            cost=0.0,
            finish_reason=str(choice.finish_reason or "stop"),
            tool_calls=tool_calls,
            json=json_parsed,
        )

    async def stream(self, req: ChatRequest) -> AsyncIterator[StreamChunk]:
        client = self._get_client()
        model = self.resolve_model(req)
        
        messages = []
        for m in self.build_messages(req):
            messages.append({
                "role": m.role,
                "content": self._map_content(m.content)
            })

        response_stream = await client.chat.stream_async(
            model=model,
            messages=messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )

        async for chunk in response_stream:
            if not chunk.data.choices:
                continue
            delta = chunk.data.choices[0].delta.content or ""
            done = chunk.data.choices[0].finish_reason is not None
            yield StreamChunk(delta=delta, done=done, provider=self.name, model=model)

    async def health_check(self) -> dict:
        start = time.time()
        try:
            client = self._get_client()
            await client.models.list_async()
            return {
                "provider": self.name,
                "status": "up",
                "latency": int((time.time() - start) * 1000),
                "checkedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        except Exception as err:
            return {
                "provider": self.name,
                "status": "down",
                "error": str(err),
                "checkedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }

    def _map_content(self, content: Union[str, List[MessageContentPart]]) -> Any:
        if isinstance(content, str):
            return content
        
        mapped = []
        for part in content:
            if part.type == "text":
                mapped.append({"type": "text", "text": part.text or ""})
            elif part.type == "image":
                mapped.append({
                    "type": "image_url",
                    "image_url": {"url": part.image.get("url") if part.image else ""}
                })
            elif part.type == "file":
                mime = part.file.get("mimeType", "") if part.file else ""
                url = part.file.get("url", "") if part.file else ""
                if mime.startswith("text/") or mime in ("application/json", "text/csv"):
                    text_val = url
                    if ";base64," in text_val:
                        parts = text_val.split(";base64,")
                        if len(parts) > 1:
                            text_val = base64.b64decode(parts[1]).decode("utf-8", errors="ignore")
                    elif not text_val.startswith("http"):
                        try:
                            text_val = base64.b64decode(text_val).decode("utf-8", errors="ignore")
                        except Exception:
                            pass
                    mapped.append({"type": "text", "text": text_val})
                else:
                    raise ValueError(
                        f"Mistral does not natively support '{mime}' file attachments."
                    )
        return mapped

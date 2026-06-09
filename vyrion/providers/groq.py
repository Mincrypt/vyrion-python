import base64
import time
from typing import Union, List, Any, AsyncIterator
from .base import BaseProvider
from ..types import ChatRequest, ChatResponse, StreamChunk, Message, MessageContentPart, TokenUsage

class GroqProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "groq"

    @property
    def default_model(self) -> str:
        return "llama-3.1-8b-instant"

    @property
    def supported_models(self) -> List[str]:
        return [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "gemma-7b-it",
            "whisper-large-v3",
        ]

    def __init__(self, config: dict):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if not self._client:
            try:
                import groq
            except ImportError:
                raise ImportError(
                    "The 'groq' package is required for GroqProvider. "
                    "Install it using 'pip install vyrion[groq]'"
                )
            self._client = groq.AsyncGroq(
                api_key=self.config.get("api_key"),
                base_url=self.config.get("base_url"),
                timeout=self.config.get("timeout") or 30.0,
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
            if req.response_format == "json":
                response_format = {"type": "json_object"}
            elif hasattr(req.response_format, "type"):
                if req.response_format.type == "json_object":
                    response_format = {"type": "json_object"}
                elif req.response_format.type == "json_schema":
                    response_format = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "response_schema",
                            "schema": req.response_format.schema,
                            "strict": True
                        }
                    }

        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            tools=tools,
            response_format=response_format,
        )

        choice = completion.choices[0]
        usage = TokenUsage(
            prompt=completion.usage.prompt_tokens if completion.usage else 0,
            completion=completion.usage.completion_tokens if completion.usage else 0,
            total=completion.usage.total_tokens if completion.usage else 0,
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
        if req.response_format and choice.message.content:
            try:
                json_parsed = json.loads(choice.message.content)
            except Exception:
                pass

        return ChatResponse(
            content=choice.message.content or "",
            provider=self.name,
            model=model,
            usage=usage,
            latency=0,
            cost=0.0,
            finish_reason=choice.finish_reason or "stop",
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

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            stream=True,
        )

        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            done = chunk.choices[0].finish_reason is not None
            yield StreamChunk(delta=delta, done=done, provider=self.name, model=model)

    async def health_check(self) -> dict:
        start = time.time()
        try:
            client = self._get_client()
            await client.models.list()
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
                        f"Groq does not natively support '{mime}' file attachments."
                    )
        return mapped

import base64
import time
from typing import Union, List, Any, AsyncIterator, Optional
from .base import BaseProvider
from ..types import ChatRequest, ChatResponse, StreamChunk, Message, MessageContentPart, TokenUsage

class AnthropicProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def default_model(self) -> str:
        return "claude-3-5-haiku-latest"

    @property
    def supported_models(self) -> List[str]:
        return [
            "claude-3-7-sonnet-latest",
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
            "claude-3-opus-latest",
            "claude-3-haiku-20240307"
        ]

    def __init__(self, config: dict):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if not self._client:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "The 'anthropic' package is required for AnthropicProvider. "
                    "Install it using 'pip install vyrion[anthropic]'"
                )
            self._client = anthropic.AsyncAnthropic(
                api_key=self.config.get("api_key"),
                base_url=self.config.get("base_url"),
                timeout=self.config.get("timeout") or 30.0,
            )
        return self._client

    async def chat(self, req: ChatRequest) -> ChatResponse:
        client = self._get_client()
        model = self.resolve_model(req)
        
        messages, system = self._build_anthropic_payload(req)

        tools = None
        if req.tools:
            tools = [{
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters or {"type": "object", "properties": {}},
            } for t in req.tools]

        if req.response_format:
            json_instruction = "IMPORTANT: You must respond ONLY with a valid JSON object."
            system = f"{system}\n\n{json_instruction}" if system else json_instruction

        response = await client.messages.create(
            model=model,
            max_tokens=req.max_tokens or 4096,
            messages=messages,
            system=system,
            temperature=req.temperature,
            tools=tools,
        )

        content_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": self._serialize_tool_input(block.input),
                    }
                })

        content = "".join(content_parts)
        usage = TokenUsage(
            prompt=response.usage.input_tokens,
            completion=response.usage.output_tokens,
            total=response.usage.input_tokens + response.usage.output_tokens,
        )

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
            finish_reason=str(response.stop_reason or "end_turn"),
            tool_calls=tool_calls if tool_calls else None,
            json=json_parsed,
        )

    async def stream(self, req: ChatRequest) -> AsyncIterator[StreamChunk]:
        client = self._get_client()
        model = self.resolve_model(req)
        messages, system = self._build_anthropic_payload(req)

        response_stream = await client.messages.create(
            model=model,
            max_tokens=req.max_tokens or 4096,
            messages=messages,
            system=system,
            temperature=req.temperature,
            stream=True,
        )

        async for event in response_stream:
            # Check event types in anthropic stream
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                yield StreamChunk(delta=event.delta.text, done=False, provider=self.name, model=model)
            elif event.type == "message_stop":
                yield StreamChunk(delta="", done=True, provider=self.name, model=model)

    async def health_check(self) -> dict:
        start = time.time()
        try:
            client = self._get_client()
            model = self.config.get("default_model") or self.default_model
            # 1 token probe check
            await client.messages.create(
                model=model,
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
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

    def _build_anthropic_payload(self, req: ChatRequest) -> tuple[List[dict], Optional[str]]:
        system = req.system_prompt
        messages = []

        for m in self.build_messages(req):
            if m.role == "system":
                if isinstance(m.content, str):
                    system = m.content
                else:
                    system = "\n".join([p.text or "" for p in m.content])
                continue

            content = m.content
            if isinstance(m.content, list):
                content = [self._map_part(p) for p in m.content]
            
            messages.append({
                "role": m.role,
                "content": content
            })

        return messages, system

    def _map_part(self, part: MessageContentPart) -> dict:
        if part.type == "text":
            return {"type": "text", "text": part.text or ""}
        elif part.type == "image":
            url = part.image.get("url", "") if part.image else ""
            media_type = part.image.get("mimeType", "image/jpeg") if part.image else "image/jpeg"
            if ";base64," in url:
                splits = url.split(";base64,")
                media_type = splits[0].split(":")[1] if len(splits[0].split(":")) > 1 else media_type
                url = splits[1]
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": url
                }
            }
        elif part.type == "file":
            url = part.file.get("url", "") if part.file else ""
            media_type = part.file.get("mimeType", "application/pdf") if part.file else "application/pdf"
            if ";base64," in url:
                splits = url.split(";base64,")
                media_type = splits[0].split(":")[1] if len(splits[0].split(":")) > 1 else media_type
                url = splits[1]

            if media_type == "application/pdf":
                return {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": url
                    }
                }
            elif media_type.startswith("text/") or media_type in ("application/json", "text/csv"):
                try:
                    text_val = base64.b64decode(url).decode("utf-8", errors="ignore")
                    return {"type": "text", "text": text_val}
                except Exception:
                    pass
            raise ValueError(
                f"Anthropic does not support '{media_type}' document format. "
                "Only PDF and text files are supported."
            )
        return {}

    def _serialize_tool_input(self, tool_input: Any) -> str:
        if isinstance(tool_input, str):
            return tool_input
        import json
        try:
            return json.dumps(tool_input)
        except Exception:
            return str(tool_input)

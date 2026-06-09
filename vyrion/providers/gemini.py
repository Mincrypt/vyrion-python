import time
from typing import Union, List, Any, AsyncIterator
from .base import BaseProvider
from ..types import ChatRequest, ChatResponse, StreamChunk, Message, MessageContentPart, TokenUsage

class GeminiProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "gemini"

    @property
    def default_model(self) -> str:
        return "gemini-2.5-flash"

    @property
    def supported_models(self) -> List[str]:
        return [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.0-pro-exp-02-05",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash"
        ]

    def __init__(self, config: dict):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if not self._client:
            try:
                from google import genai
            except ImportError:
                raise ImportError(
                    "The 'google-genai' package is required for GeminiProvider. "
                    "Install it using 'pip install vyrion[gemini]'"
                )
            self._client = genai.Client(api_key=self.config.get("api_key"))
        return self._client

    async def chat(self, req: ChatRequest) -> ChatResponse:
        client = self._get_client()
        model = self.resolve_model(req)
        
        # Build contents list
        contents = []
        for m in self.build_messages(req):
            if m.role == "system":
                continue  # system prompt is passed as systemInstruction config
            
            parts = []
            if isinstance(m.content, str):
                parts = [{"text": m.content}]
            else:
                parts = [self._map_part(p) for p in m.content]
            
            contents.append({
                "role": "model" if m.role == "assistant" else "user",
                "parts": parts
            })

        # Process config params
        config_params = {}
        if req.max_tokens:
            config_params["max_output_tokens"] = req.max_tokens
        if req.temperature:
            config_params["temperature"] = req.temperature
        if req.system_prompt:
            config_params["system_instruction"] = req.system_prompt

        # Tools & formatting
        if req.tools:
            config_params["tools"] = [{
                "function_declarations": [{
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                } for t in req.tools]
            }]

        if req.response_format:
            config_params["response_mime_type"] = "application/json"
            if hasattr(req.response_format, "schema") and req.response_format.schema:
                config_params["response_schema"] = req.response_format.schema

        # Execute call inside loop (since Python client generate_content is synchronous, we run in executor or if it is async)
        # Note: the new google-genai Client supports synchronous methods, but we can call it in loop or use client.aio for async!
        # Yes, Google GenAI SDK has an async client: `client.aio.models.generate_content`!
        # This is async and perfect!
        result = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config_params,
        )

        text = result.text or ""
        usage = TokenUsage(
            prompt=result.usage_metadata.prompt_token_count if result.usage_metadata else 0,
            completion=result.usage_metadata.candidates_token_count if result.usage_metadata else 0,
            total=result.usage_metadata.total_token_count if result.usage_metadata else 0,
        )

        tool_calls = None
        if hasattr(result, "function_calls") and result.function_calls:
            import json
            tool_calls = [{
                "id": f"call_{fc.name}_{i}",
                "type": "function",
                "function": {
                    "name": fc.name,
                    "arguments": json.dumps(fc.args),
                }
            } for i, fc in enumerate(result.function_calls)]

        import json
        json_parsed = None
        if req.response_format and text:
            try:
                json_parsed = json.loads(text)
            except Exception:
                pass

        finish_reason = "stop"
        if result.candidates and result.candidates[0].finish_reason:
            finish_reason = str(result.candidates[0].finish_reason)

        return ChatResponse(
            content=text,
            provider=self.name,
            model=model,
            usage=usage,
            latency=0,
            cost=0.0,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            json=json_parsed,
        )

    async def stream(self, req: ChatRequest) -> AsyncIterator[StreamChunk]:
        client = self._get_client()
        model = self.resolve_model(req)
        
        contents = []
        for m in self.build_messages(req):
            if m.role == "system":
                continue
            parts = []
            if isinstance(m.content, str):
                parts = [{"text": m.content}]
            else:
                parts = [self._map_part(p) for p in m.content]
            contents.append({
                "role": "model" if m.role == "assistant" else "user",
                "parts": parts
            })

        config_params = {}
        if req.max_tokens:
            config_params["max_output_tokens"] = req.max_tokens
        if req.temperature:
            config_params["temperature"] = req.temperature
        if req.system_prompt:
            config_params["system_instruction"] = req.system_prompt

        response_stream = await client.aio.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config_params,
        )

        async for chunk in response_stream:
            delta = chunk.text or ""
            done = False
            if chunk.candidates and chunk.candidates[0].finish_reason:
                done = str(chunk.candidates[0].finish_reason) != "FINISH_REASON_UNSPECIFIED"
            yield StreamChunk(delta=delta, done=done, provider=self.name, model=model)

    async def health_check(self) -> dict:
        start = time.time()
        try:
            client = self._get_client()
            model = self.config.get("default_model") or self.default_model
            await client.aio.models.get(model=model)
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

    def _map_part(self, part: MessageContentPart) -> dict:
        if part.type == "text":
            return {"text": part.text or ""}
        elif part.type == "image":
            url = part.image.get("url", "") if part.image else ""
            mime = part.image.get("mimeType", "image/jpeg") if part.image else "image/jpeg"
            if ";base64," in url:
                splits = url.split(";base64,")
                mime = splits[0].split(":")[1] if len(splits[0].split(":")) > 1 else mime
                url = splits[1]
            return {"inline_data": {"data": url, "mime_type": mime}}
        elif part.type == "file":
            url = part.file.get("url", "") if part.file else ""
            mime = part.file.get("mimeType", "application/pdf") if part.file else "application/pdf"
            if ";base64," in url:
                splits = url.split(";base64,")
                mime = splits[0].split(":")[1] if len(splits[0].split(":")) > 1 else mime
                url = splits[1]
            return {"inline_data": {"data": url, "mime_type": mime}}
        return {}

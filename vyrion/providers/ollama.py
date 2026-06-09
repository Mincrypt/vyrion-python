import time
import json
import base64
import aiohttp
from typing import Union, List, Any, AsyncIterator
from .base import BaseProvider
from ..types import ChatRequest, ChatResponse, StreamChunk, Message, MessageContentPart, TokenUsage

DEFAULT_OLLAMA_BASE = "http://localhost:11434"

class OllamaProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "ollama"

    @property
    def default_model(self) -> str:
        return "llama3.2"

    @property
    def supported_models(self) -> List[str]:
        return [
            "llama3.2",
            "llama3.2:1b",
            "llama3.1",
            "llama3.1:70b",
            "phi4",
            "phi3",
            "mistral",
            "mistral-nemo",
            "gemma3",
            "qwen2.5",
            "deepseek-r1",
            "codellama",
            "nomic-embed-text",
        ]

    def __init__(self, config: dict):
        super().__init__(config)

    @property
    def base_url(self) -> str:
        url = self.config.get("base_url") or DEFAULT_OLLAMA_BASE
        return url.rstrip("/")

    def is_available(self) -> bool:
        # Always available to attempt; health check determines actual status
        return True

    async def chat(self, req: ChatRequest) -> ChatResponse:
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
            response_format = "json"

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "tools": tools,
            "format": response_format,
            "options": {}
        }
        if req.max_tokens:
            payload["options"]["num_predict"] = req.max_tokens
        if req.temperature:
            payload["options"]["temperature"] = req.temperature

        start = time.time()
        timeout = aiohttp.ClientTimeout(total=self.config.get("timeout") or 30.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{self.base_url}/api/chat", json=payload) as resp:
                if resp.status != 200:
                    err_text = await resp.text()
                    raise RuntimeError(f"Ollama error {resp.status}: {err_text}")
                
                data = await resp.json()

        latency = int((time.time() - start) * 1000)
        msg = data.get("message", {})
        content = msg.get("content", "")
        
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        usage = TokenUsage(
            prompt=prompt_tokens,
            completion=completion_tokens,
            total=prompt_tokens + completion_tokens
        )

        tool_calls = None
        raw_tool_calls = msg.get("tool_calls")
        if raw_tool_calls:
            tool_calls = []
            for i, tc in enumerate(raw_tool_calls):
                fn = tc.get("function", {})
                args = fn.get("arguments", {})
                if not isinstance(args, str):
                    args = json.dumps(args)
                tool_calls.append({
                    "id": tc.get("id") or f"call_{fn.get('name')}_{i}",
                    "type": "function",
                    "function": {
                        "name": fn.get("name"),
                        "arguments": args
                    }
                })

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
            latency=latency,
            cost=0.0,
            finish_reason="stop" if data.get("done") else "length",
            tool_calls=tool_calls,
            json=json_parsed,
        )

    async def stream(self, req: ChatRequest) -> AsyncIterator[StreamChunk]:
        model = self.resolve_model(req)
        messages = []
        for m in self.build_messages(req):
            messages.append({
                "role": m.role,
                "content": self._map_content(m.content)
            })

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {}
        }
        if req.max_tokens:
            payload["options"]["num_predict"] = req.max_tokens
        if req.temperature:
            payload["options"]["temperature"] = req.temperature

        timeout = aiohttp.ClientTimeout(total=self.config.get("timeout") or 30.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{self.base_url}/api/chat", json=payload) as resp:
                if resp.status != 200:
                    err_text = await resp.text()
                    raise RuntimeError(f"Ollama stream error {resp.status}: {err_text}")
                
                async for line in resp.content:
                    if not line:
                        continue
                    try:
                        chunk_data = json.loads(line.decode("utf-8").strip())
                        delta = chunk_data.get("message", {}).get("content", "")
                        done = chunk_data.get("done", False)
                        yield StreamChunk(delta=delta, done=done, provider=self.name, model=model)
                    except Exception:
                        pass

    async def health_check(self) -> dict:
        start = time.time()
        try:
            timeout = aiohttp.ClientTimeout(total=5.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{self.base_url}/api/tags") as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"HTTP {resp.status}")
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
        
        # Ollama expects simple string contents for messages, or text representation
        text_parts = []
        for part in content:
            if part.type == "text":
                text_parts.append(part.text or "")
            elif part.type == "file":
                mime = part.file.get("mimeType", "") if part.file else ""
                url = part.file.get("url", "") if part.file else ""
                if mime.startswith("text/") or mime in ("application/json", "text/csv"):
                    text_val = url
                    if ";base64," in text_val:
                        splits = text_val.split(";base64,")
                        if len(splits) > 1:
                            text_val = base64.b64decode(splits[1]).decode("utf-8", errors="ignore")
                    elif not text_val.startswith("http"):
                        try:
                            text_val = base64.b64decode(text_val).decode("utf-8", errors="ignore")
                        except Exception:
                            pass
                    text_parts.append(text_val)
        return "\n".join(text_parts)

import time
import json
from typing import Optional, Dict, Any
from ..types import ChatRequest

class InMemoryCache:
    def __init__(self):
        self._store: Dict[str, tuple[Any, Optional[float]]] = {}

    def get(self, key: str) -> Optional[Any]:
        if key not in self._store:
            return None
        value, expires_at = self._store[key]
        if expires_at is not None and time.time() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        expires_at = time.time() + ttl if ttl is not None else None
        self._store[key] = (value, expires_at)

    def delete(self, key: str) -> None:
        if key in self._store:
            del self._store[key]

    def clear(self) -> None:
        self._store.clear()

def generate_cache_key(req: ChatRequest) -> str:
    messages_serialized = []
    if req.messages:
        for m in req.messages:
            # content can be string or list of MessageContentPart
            if isinstance(m.content, str):
                messages_serialized.append(f"{m.role}:{m.content}")
            else:
                parts_str = ",".join([f"{p.type}:{p.text or ''}" for p in m.content])
                messages_serialized.append(f"{m.role}:{parts_str}")

    key_parts = {
        "message": req.message,
        "messages": messages_serialized if req.messages else None,
        "system_prompt": req.system_prompt,
        "provider": req.provider,
        "model": req.model,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "goal": req.goal if isinstance(req.goal, str) else None,
    }
    return json.dumps(key_parts, sort_keys=True)

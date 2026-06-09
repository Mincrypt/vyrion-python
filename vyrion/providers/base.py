from abc import ABC, abstractmethod
from typing import List, Dict, Any, AsyncIterator
from ..types import ChatRequest, ChatResponse, StreamChunk, Message

class BaseProvider(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        pass

    @property
    @abstractmethod
    def supported_models(self) -> List[str]:
        pass

    def is_available(self) -> bool:
        return bool(self.config.get("api_key") or self.config.get("base_url"))

    @abstractmethod
    async def chat(self, req: ChatRequest) -> ChatResponse:
        pass

    @abstractmethod
    async def stream(self, req: ChatRequest) -> AsyncIterator[StreamChunk]:
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        pass

    def resolve_model(self, req: ChatRequest) -> str:
        return req.model or self.config.get("default_model") or self.default_model

    def build_messages(self, req: ChatRequest) -> List[Message]:
        if req.messages:
            return req.messages
        
        msgs = []
        if req.system_prompt:
            msgs.append(Message(role="system", content=req.system_prompt))
        if req.message:
            msgs.append(Message(role="user", content=req.message))
        return msgs

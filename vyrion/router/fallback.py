import time
from typing import Dict, List, Any, Optional, AsyncIterator
from ..types import ChatRequest, ChatResponse, StreamChunk
from .strategies import resolve_strategy
from .circuit import CircuitBreakerManager

DEFAULT_FALLBACK = [
    "openai",
    "groq",
    "gemini",
    "anthropic",
    "mistral",
    "together",
    "ollama",
]

class FallbackRouter:
    def __init__(
        self,
        providers: Dict[str, Any],
        analytics: Any,
        circuit: Optional[CircuitBreakerManager] = None,
    ):
        self.providers = providers
        self.analytics = analytics
        self.circuit = circuit

    async def chat(self, req: ChatRequest) -> ChatResponse:
        chain = self.build_chain(req)
        last_error = None

        for provider in chain:
            try:
                start = time.time()
                response = await provider.chat(req)
                latency = int((time.time() - start) * 1000)
                
                response.latency = latency
                if response.cost == 0.0:
                    from ..analytics.cost import estimate_cost
                    response.cost = estimate_cost(response.provider, response.model, response.usage)
                if self.analytics:
                    self.analytics.record(
                        provider=provider.name,
                        latency=latency,
                        tokens=response.usage.total,
                        cost=response.cost,
                        success=True,
                    )
                if self.circuit:
                    self.circuit.record_success(provider.name)
                return response
            except Exception as err:
                last_error = err
                if self.analytics:
                    self.analytics.record(
                        provider=provider.name,
                        latency=0,
                        tokens=0,
                        cost=0.0,
                        success=False,
                    )
                if self.circuit:
                    self.circuit.record_failure(provider.name, err)

        tried_list = " → ".join([p.name for p in chain])
        raise RuntimeError(
            f"All configured providers failed. Last error: {last_error}\n"
            f"Providers tried: {tried_list}"
        )

    async def stream(self, req: ChatRequest) -> AsyncIterator[StreamChunk]:
        chain = self.build_chain(req)
        last_error = None

        for provider in chain:
            try:
                start = time.time()
                tokens = 0
                
                async for chunk in provider.stream(req):
                    tokens += len(chunk.delta)
                    yield chunk

                latency = int((time.time() - start) * 1000)
                if self.analytics:
                    self.analytics.record(
                        provider=provider.name,
                        latency=latency,
                        tokens=tokens,
                        cost=0.0,
                        success=True,
                    )
                if self.circuit:
                    self.circuit.record_success(provider.name)
                return
            except Exception as err:
                last_error = err
                if self.analytics:
                    self.analytics.record(
                        provider=provider.name,
                        latency=0,
                        tokens=0,
                        cost=0.0,
                        success=False,
                    )
                if self.circuit:
                    self.circuit.record_failure(provider.name, err)

        tried_list = " → ".join([p.name for p in chain])
        raise RuntimeError(
            f"All configured providers failed during streaming. Last error: {last_error}\n"
            f"Providers tried: {tried_list}"
        )

    def build_chain(self, req: ChatRequest) -> List[Any]:
        all_available = [p for p in self.providers.values() if p.is_available()]
        if not all_available:
            raise RuntimeError(
                "No active providers found. Please configure at least one provider API key."
            )

        available = all_available
        if self.circuit:
            active = [p for p in all_available if self.circuit.is_available(p.name)]
            if active:
                available = active

        if req.provider and req.provider != "auto":
            primary = self.providers.get(req.provider)
            if not primary or not primary.is_available():
                raise ValueError(f"Provider '{req.provider}' is not configured.")

            fallback_names = req.fallback if req.fallback is not None else DEFAULT_FALLBACK
            rest = [
                self.providers[n]
                for n in fallback_names
                if n != req.provider and n in self.providers and self.providers[n].is_available()
            ]

            active_rest = rest
            cooldown_rest = []
            if self.circuit:
                active_rest = [p for p in rest if self.circuit.is_available(p.name)]
                cooldown_rest = [p for p in rest if not self.circuit.is_available(p.name)]

            if self.circuit and not self.circuit.is_available(primary.name):
                if active_rest:
                    return active_rest + [primary] + cooldown_rest

            return [primary] + active_rest + cooldown_rest

        goal = req.goal or "auto"
        strategy = req.goal if callable(req.goal) else resolve_strategy(goal)
        primary = strategy(available, self.analytics)

        fallback_names = req.fallback if req.fallback is not None else DEFAULT_FALLBACK
        rest = [
            self.providers[n]
            for n in fallback_names
            if n != primary.name and n in self.providers and self.providers[n].is_available()
        ]

        active_rest = rest
        cooldown_rest = []
        if self.circuit:
            active_rest = [p for p in rest if self.circuit.is_available(p.name)]
            cooldown_rest = [p for p in rest if not self.circuit.is_available(p.name)]

        in_chain = {primary.name} | {p.name for p in active_rest} | {p.name for p in cooldown_rest}
        extra = [p for p in available if p.name not in in_chain]

        return [primary] + active_rest + cooldown_rest + extra

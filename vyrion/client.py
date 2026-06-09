import asyncio
from typing import List, Dict, Any, Union, Optional, AsyncIterator, Callable
from .types import ChatRequest, ChatResponse, StreamChunk, Message, MessageContentPart, ProviderConfig, AnalyticsSnapshot, HealthCheckResult
from .router.fallback import FallbackRouter
from .router.circuit import CircuitBreakerManager
from .cache.memory import InMemoryCache, generate_cache_key
from .middleware.compose import compose

from .providers.openai import OpenAIProvider
from .providers.gemini import GeminiProvider
from .providers.anthropic import AnthropicProvider
from .providers.groq import GroqProvider
from .providers.together import TogetherProvider
from .providers.mistral import MistralProvider
from .providers.ollama import OllamaProvider

from .analytics.tracker import AnalyticsTracker
from .analytics.health import HealthMonitor
from .analytics.cost import estimate_cost, get_pricing, set_pricing

class Vyrion:
    def __init__(self, **config):
        self.config = config
        self.providers: Dict[str, Any] = {}
        self.middlewares: List[Callable] = []
        
        self.timeout = config.get("timeout")
        self.fallback = config.get("fallback")
        self.default_goal = config.get("default_goal", "auto")
        
        self.cache = None
        cache_config = config.get("cache", False)
        if cache_config:
            self.cache = cache_config if not isinstance(cache_config, bool) else InMemoryCache()
            self.middlewares.append(self._create_cache_middleware(self.cache))

        self.circuit_breaker = None
        cb_config = config.get("circuit_breaker")
        if cb_config:
            if isinstance(cb_config, dict):
                self.circuit_breaker = CircuitBreakerManager(
                    failures_threshold=cb_config.get("failures_threshold", 3),
                    cooldown_seconds=cb_config.get("cooldown_seconds", 60)
                )
            else:
                self.circuit_breaker = cb_config

        self.analytics = AnalyticsTracker()
        self.health = HealthMonitor()
        self._register_builtins()

        # Raise error if no active providers configured
        if not self.providers:
            raise ValueError(
                "No active providers found. Please configure at least one provider API key.\n\n"
                "Examples:\n"
                "  ai = Vyrion(openai='your-openai-api-key')\n"
                "  ai = Vyrion(groq='your-groq-api-key')\n"
                "  ai = Vyrion(ollama={})\n"
            )

        self.router = FallbackRouter(self.providers, self.analytics, self.circuit_breaker)

    def use(self, middleware: Callable) -> None:
        self.middlewares.append(middleware)

    def register_provider(self, provider: Any) -> None:
        self.providers[provider.name] = provider

    def unregister_provider(self, name: str) -> bool:
        if name in self.providers:
            del self.providers[name]
            return True
        return False

    def get_providers(self) -> List[str]:
        return list(self.providers.keys())

    def get_available_providers(self) -> List[str]:
        return [name for name, p in self.providers.items() if p.is_available()]

    # ── Analytics API ─────────────────────────────────────────

    def get_stats(self) -> AnalyticsSnapshot:
        return self.analytics.get_snapshot()

    def reset_stats(self) -> None:
        self.analytics.reset()

    def get_total_cost(self) -> float:
        return self.analytics.get_snapshot().total_cost

    # ── Health API ────────────────────────────────────────────

    async def get_provider_health(self) -> List[HealthCheckResult]:
        self.health.providers = list(self.providers.values())
        return await self.health.check_all()

    def start_health_monitor(self, interval_seconds: int = 300) -> None:
        self.health.interval_seconds = interval_seconds
        self.health.start(list(self.providers.values()))

    def stop_health_monitor(self) -> None:
        self.health.stop()

    # ── Pricing API ───────────────────────────────────────────

    def get_pricing(self, provider: Optional[str] = None) -> Any:
        return get_pricing(provider)

    def set_pricing(self, provider: str, model: str, pricing: dict) -> None:
        set_pricing(provider, model, pricing)

    # ── Execution API ─────────────────────────────────────────

    async def chat(self, request_payload: Union[dict, ChatRequest]) -> ChatResponse:
        req = self._coerce_request(request_payload)

        async def core_executor(ctx, next_dispatch=None):
            response = await self.router.chat(ctx["request"])
            if response.cost == 0.0:
                response.cost = estimate_cost(response.provider, response.model, response.usage)
            return response

        context = {"request": req}
        if not self.middlewares:
            return await core_executor(context)

        runner = compose(self.middlewares)
        return await runner(context, core_executor)

    async def stream(self, request_payload: Union[dict, ChatRequest]) -> AsyncIterator[StreamChunk]:
        req = self._coerce_request(request_payload)
        
        cache_enabled = self.cache is not None and req.cache
        if cache_enabled:
            key = "stream:" + generate_cache_key(req)
            try:
                cached = self.cache.get(key)
                if cached and isinstance(cached, list):
                    for chunk in cached:
                        yield chunk
                        await asyncio.sleep(0.015)
                    return
            except Exception:
                pass

            chunks = []
            try:
                async for chunk in self.router.stream(req):
                    chunks.append(chunk)
                    yield chunk
                try:
                    self.cache.set(key, chunks)
                except Exception:
                    pass
            except Exception as err:
                raise err
        else:
            async for chunk in self.router.stream(req):
                yield chunk

    def _coerce_request(self, payload: Union[dict, ChatRequest]) -> ChatRequest:
        if isinstance(payload, dict):
            messages = None
            if "messages" in payload:
                messages = []
                for m in payload["messages"]:
                    content = m["content"]
                    if isinstance(content, list):
                        content = [
                            MessageContentPart(
                                type=p.get("type"),
                                text=p.get("text"),
                                image=p.get("image"),
                                file=p.get("file"),
                            ) for p in content
                        ]
                    messages.append(Message(role=m["role"], content=content))

            return ChatRequest(
                message=payload.get("message"),
                messages=messages,
                system_prompt=payload.get("system_prompt") or payload.get("systemPrompt"),
                provider=payload.get("provider", "auto"),
                model=payload.get("model"),
                goal=payload.get("goal", self.default_goal),
                fallback=payload.get("fallback", self.fallback),
                max_tokens=payload.get("max_tokens") or payload.get("maxTokens"),
                temperature=payload.get("temperature"),
                stream=payload.get("stream", False),
                cache=payload.get("cache", True),
                tools=payload.get("tools"),
                response_format=payload.get("response_format") or payload.get("responseFormat"),
            )
        
        if payload.goal == "auto":
            payload.goal = self.default_goal
        if payload.fallback is None:
            payload.fallback = self.fallback
        return payload

    def _register_builtins(self) -> None:
        timeout = self.timeout
        builtins = {
            "openai": OpenAIProvider,
            "gemini": GeminiProvider,
            "anthropic": AnthropicProvider,
            "groq": GroqProvider,
            "together": TogetherProvider,
            "mistral": MistralProvider,
            "ollama": OllamaProvider,
        }

        for name, provider_class in builtins.items():
            conf_val = self.config.get(name)
            if conf_val is not None:
                p_conf = {}
                if isinstance(conf_val, str):
                    p_conf = {"api_key": conf_val, "timeout": timeout}
                elif isinstance(conf_val, dict):
                    p_conf = {"timeout": timeout, **conf_val}
                
                api_key = p_conf.get("api_key") or p_conf.get("apiKey")
                base_url = p_conf.get("base_url") or p_conf.get("baseUrl")
                default_model = p_conf.get("default_model") or p_conf.get("defaultModel")
                
                self.providers[name] = provider_class({
                    "api_key": api_key,
                    "base_url": base_url,
                    "default_model": default_model,
                    "timeout": p_conf.get("timeout")
                })

    def _create_cache_middleware(self, cache_instance: Any) -> Callable:
        async def cache_middleware(ctx: dict, next_dispatch: Callable) -> Any:
            req = ctx["request"]
            if req.stream or not req.cache:
                return await next_dispatch()

            key = generate_cache_key(req)
            try:
                cached = cache_instance.get(key)
                if cached:
                    return cached
            except Exception:
                pass

            response = await next_dispatch()
            try:
                cache_instance.set(key, response)
            except Exception:
                pass
            return response
        return cache_middleware

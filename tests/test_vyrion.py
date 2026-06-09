import asyncio
import time
import pytest
from typing import AsyncIterator, Any, List
from vyrion import Vyrion, ChatResponse, StreamChunk, InMemoryCache
from vyrion.types import TokenUsage, Message, MessageContentPart, ChatRequest

# ── Helper Mocks ──────────────────────────────────────────

class MockProvider:
    def __init__(self, name: str, fails_times: int = 0, rate_limit: bool = False):
        self.name = name
        self.default_model = "mock-model"
        self.supported_models = ["mock-model"]
        self.fails_times = fails_times
        self.rate_limit = rate_limit
        self.calls = 0

    def is_available(self) -> bool:
        return True

    async def chat(self, req: ChatRequest) -> ChatResponse:
        self.calls += 1
        if self.fails_times > 0 and self.calls <= self.fails_times:
            if self.rate_limit:
                # Mock HTTP 429 Rate Limit
                err = RuntimeError("429 Rate Limit Exceeded")
                err.status = 429
                raise err
            raise RuntimeError("Temporary provider error")
            
        return ChatResponse(
            content=f"Mock response from {self.name}",
            provider=self.name,
            model=self.default_model,
            usage=TokenUsage(10, 10, 20),
            latency=5,
            cost=0.0
        )

    async def stream(self, req: ChatRequest) -> AsyncIterator[StreamChunk]:
        self.calls += 1
        if self.fails_times > 0 and self.calls <= self.fails_times:
            raise RuntimeError("Stream failed")
            
        yield StreamChunk(delta=f"Hello from {self.name}", done=False, provider=self.name, model=self.default_model)
        yield StreamChunk(delta=f" stream", done=True, provider=self.name, model=self.default_model)

    async def health_check(self) -> dict:
        return {}


# ── Testing Suites ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_simple_routing_and_failover():
    """Verify router fails over to secondary providers when primary fails."""
    ai = Vyrion()
    
    bad_p = MockProvider("bad-provider", fails_times=1)
    good_p = MockProvider("good-provider")
    
    ai.register_provider(bad_p)
    ai.register_provider(good_p)

    res = await ai.chat({
        "message": "hello",
        "provider": "bad-provider",
        "fallback": ["bad-provider", "good-provider"]
    })

    assert res.content == "Mock response from good-provider"
    assert bad_p.calls == 1
    assert good_p.calls == 1


@pytest.mark.asyncio
async def test_circuit_breaker_threshold_trip():
    """Verify circuit breaker trips and bypasses degraded provider after sequential errors."""
    ai = Vyrion(
        circuit_breaker={"failures_threshold": 2, "cooldown_seconds": 5}
    )
    
    bad_p = MockProvider("bad-provider", fails_times=5)
    good_p = MockProvider("good-provider")
    
    ai.register_provider(bad_p)
    ai.register_provider(good_p)

    # Call 1: bad-provider fails (calls = 1) -> tries good-provider (succeeds)
    res = await ai.chat({
        "message": "hello",
        "provider": "bad-provider",
        "fallback": ["bad-provider", "good-provider"]
    })
    assert res.content == "Mock response from good-provider"
    assert bad_p.calls == 1

    # Call 2: bad-provider fails again (calls = 2) -> TRIPS circuit! tries good-provider (succeeds)
    res = await ai.chat({
        "message": "hello",
        "provider": "bad-provider",
        "fallback": ["bad-provider", "good-provider"]
    })
    assert res.content == "Mock response from good-provider"
    assert bad_p.calls == 2

    # Call 3: bad-provider is on cooldown! Bypassed completely -> routes to good-provider directly
    res = await ai.chat({
        "message": "hello",
        "provider": "bad-provider",
        "fallback": ["bad-provider", "good-provider"]
    })
    assert res.content == "Mock response from good-provider"
    assert bad_p.calls == 2  # Remains 2! Bypassed!


@pytest.mark.asyncio
async def test_circuit_breaker_instant_429_trip():
    """Verify circuit breaker trips instantly on HTTP 429 rate limit exceptions."""
    ai = Vyrion(
        circuit_breaker={"failures_threshold": 5, "cooldown_seconds": 5}
    )
    
    rate_limit_p = MockProvider("rl-provider", fails_times=1, rate_limit=True)
    good_p = MockProvider("good-provider")
    
    ai.register_provider(rate_limit_p)
    ai.register_provider(good_p)

    # Call 1: rl-provider throws 429 -> trips circuit breaker instantly -> tries good-provider (succeeds)
    res = await ai.chat({
        "message": "hello",
        "provider": "rl-provider",
        "fallback": ["rl-provider", "good-provider"]
    })
    assert res.content == "Mock response from good-provider"
    assert rate_limit_p.calls == 1

    # Call 2: rl-provider is on cooldown -> bypassed immediately
    res = await ai.chat({
        "message": "hello",
        "provider": "rl-provider",
        "fallback": ["rl-provider", "good-provider"]
    })
    assert res.content == "Mock response from good-provider"
    assert rate_limit_p.calls == 1  # Remains 1!


@pytest.mark.asyncio
async def test_chat_and_stream_caching():
    """Verify chat and stream requests are cached and play back correctly."""
    ai = Vyrion(cache=True)
    
    stream_provider = MockProvider("stream-provider")
    ai.register_provider(stream_provider)

    # 1. Chat Caching
    res1 = await ai.chat({"message": "test-chat", "provider": "stream-provider"})
    res2 = await ai.chat({"message": "test-chat", "provider": "stream-provider"})
    assert res1.content == res2.content
    assert stream_provider.calls == 1  # Called only once

    # 2. Stream Caching
    chunks1 = []
    async for chunk in ai.stream({"message": "test-stream", "provider": "stream-provider"}):
        chunks1.append(chunk)

    assert len(chunks1) == 2
    assert chunks1[0].delta == "Hello from stream-provider"
    assert stream_provider.calls == 2  # Incremented to 2 for the first stream

    # Fetch stream again (cache hit)
    start = time.time()
    chunks2 = []
    async for chunk in ai.stream({"message": "test-stream", "provider": "stream-provider"}):
        chunks2.append(chunk)

    elapsed_ms = int((time.time() - start) * 1000)
    assert len(chunks2) == 2
    assert chunks2[0].delta == "Hello from stream-provider"
    assert stream_provider.calls == 2  # Still 2 (no call to provider stream)
    # Timing playback delay check (since 2 chunks with 15ms sleep = ~15ms - 30ms elapsed)
    assert elapsed_ms >= 10


@pytest.mark.asyncio
async def test_onion_middleware_lifecycle():
    """Verify onion middleware runs in wrapping lifecycle sequences."""
    ai = Vyrion()
    
    mock_p = MockProvider("provider-a")
    ai.register_provider(mock_p)

    order = []

    async def mw_one(ctx, next_dispatch):
        order.append("one_in")
        res = await next_dispatch()
        order.append("one_out")
        return res

    async def mw_two(ctx, next_dispatch):
        order.append("two_in")
        res = await next_dispatch()
        order.append("two_out")
        return res

    ai.use(mw_one)
    ai.use(mw_two)

    await ai.chat({"message": "mw-test", "provider": "provider-a"})
    
    assert order == ["one_in", "two_in", "two_out", "one_out"]


@pytest.mark.asyncio
async def test_multimodal_openai_unsupported_document():
    """Verify OpenAI provider raises error on unsupported binary document parts."""
    from vyrion.providers.openai import OpenAIProvider
    
    prov = OpenAIProvider({"api_key": "test"})
    req = ChatRequest(
        messages=[
            Message(
                role="user",
                content=[
                    MessageContentPart(
                        type="file",
                        file={"url": "data:application/pdf;base64,pdfdata...", "mimeType": "application/pdf"}
                    )
                ]
            )
        ]
    )

    with pytest.raises(ValueError, match="OpenAI does not natively support 'application/pdf'"):
        prov._map_content(req.messages[0].content)

import asyncio
import time
from typing import AsyncIterator, Any
from vyrion import Vyrion, ChatResponse, StreamChunk

async def example_simple_chat(ai: Vyrion):
    print("\n-- 1. Simple Chat (auto routing) -----------------")
    # Register mock providers so it always runs even without keys
    ai.register_provider(MockChatProvider("openai"))
    ai.register_provider(MockChatProvider("gemini"))

    res = await ai.chat({
        "message": "What is recursion? Explain in 2 sentences.",
        "provider": "auto"
    })
    print(f"[{res.provider} / {res.model}] {res.content}")
    print(f"Tokens: {res.usage.total} | Latency: {res.latency}ms")

async def example_streaming(ai: Vyrion):
    print("\n-- 2. Streaming ---------------------------------")
    ai.register_provider(MockChatProvider("groq"))

    print("[stream] ", end="", flush=True)
    async for chunk in ai.stream({
        "message": "Tell me a fun fact about the universe.",
        "provider": "groq"
    }):
        print(chunk.delta, end="", flush=True)
    print()

async def example_circuit_breaker():
    print("\n-- 3. Circuit Breaker & Cooldown --------------")
    circuit_client = Vyrion(
        openai="invalid-key-to-force-fail",
        circuit_breaker={
            "failures_threshold": 2,
            "cooldown_seconds": 3
        }
    )

    print("Making request 1 (fails)...")
    try:
        await circuit_client.chat({"provider": "openai", "message": "Hello"})
    except Exception:
        print("Request 1 failed as expected.")

    print("Making request 2 (fails -> trips circuit)...")
    try:
        await circuit_client.chat({"provider": "openai", "message": "Hello"})
    except Exception:
        print("Request 2 failed as expected. Circuit tripped!")

    print("Making request 3 (instantly bypassed!)...")
    try:
        await circuit_client.chat({"provider": "openai", "message": "Hello"})
    except Exception as err:
        print(f"Request 3 failed instantly: {err}")

async def example_streaming_cache():
    print("\n-- 4. Streaming Caching & Playback -----------")
    cached_client = Vyrion(
        openai="sk-test",
        cache=True
    )

    class CountingStreamProvider:
        name = "counting-stream"
        default_model = "model-s"
        supported_models = ["model-s"]
        
        def __init__(self):
            self.calls = 0

        def is_available(self) -> bool:
            return True

        async def chat(self, req) -> ChatResponse:
            return ChatResponse(content="result", provider=self.name, model=self.default_model, usage=None, latency=5)

        async def stream(self, req) -> AsyncIterator[StreamChunk]:
            self.calls += 1
            yield StreamChunk(delta="Hello", done=False, provider=self.name, model=self.default_model)
            yield StreamChunk(delta=" from", done=False, provider=self.name, model=self.default_model)
            yield StreamChunk(delta=" python", done=False, provider=self.name, model=self.default_model)
            yield StreamChunk(delta=" stream", done=False, provider=self.name, model=self.default_model)
            yield StreamChunk(delta=" cache!", done=True, provider=self.name, model=self.default_model)

        async def health_check(self) -> dict:
            return {
                "provider": self.name,
                "status": "up",
                "latency": 5,
                "checkedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }

    mock_provider = CountingStreamProvider()
    cached_client.register_provider(mock_provider)

    print("Stream 1 (Cache Miss - Live):")
    print("[live] ", end="", flush=True)
    async for chunk in cached_client.stream({"message": "test", "provider": "counting-stream"}):
        print(chunk.delta, end="", flush=True)
    print()

    print("\nStream 2 (Cache Hit - Playback with 15ms delay):")
    start = time.time()
    print("[playback] ", end="", flush=True)
    async for chunk in cached_client.stream({"message": "test", "provider": "counting-stream"}):
        print(chunk.delta, end="", flush=True)
    print(f"\n(Playback took {int((time.time() - start) * 1000)}ms)")
    print(f"Provider calls: {mock_provider.calls}")


class MockChatProvider:
    def __init__(self, name: str):
        self.name = name
        self.default_model = "mock-model"
        self.supported_models = ["mock-model"]

    def is_available(self) -> bool:
        return True

    async def chat(self, req) -> ChatResponse:
        from vyrion.types import TokenUsage
        return ChatResponse(
            content=f"Mock response from {self.name}",
            provider=self.name,
            model=self.default_model,
            usage=TokenUsage(10, 10, 20),
            latency=10,
            cost=0.0
        )

    async def stream(self, req) -> Any:
        yield StreamChunk(delta=f"Mock stream content from {self.name}", done=True, provider=self.name, model=self.default_model)

    async def health_check(self) -> dict:
        return {
            "provider": self.name,
            "status": "up",
            "latency": 5,
            "checkedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }


async def example_analytics_and_health(ai: Vyrion):
    print("\n-- 5. Analytics & Health Checks ---------------")
    # Expose registered providers list
    print(f"Registered providers: {ai.get_providers()}")
    print(f"Available providers: {ai.get_available_providers()}")
    
    # Retrieve current stats snapshots
    stats = ai.get_stats()
    print(f"Total requests recorded: {stats.total_requests}")
    print(f"Total USD cost: ${stats.total_cost:.6f}")
    for ps in stats.providers:
        print(f"  - {ps.provider}: {ps.requests} requests | Avg Latency: {ps.avg_latency}ms | Cost: ${ps.total_cost:.6f}")

    # Trigger health checks
    print("Running immediate health checks...")
    healths = await ai.get_provider_health()
    for h in healths:
        icon = "[OK]" if h.status == "up" else "[FAIL]"
        print(f"  {icon} {h.provider}: {h.status} ({h.latency}ms)")


async def main():
    ai = Vyrion(openai="mock-key")
    # Configure custom pricing overrides
    ai.set_pricing("openai", "mock-model", {"inputPer1M": 1.5, "outputPer1M": 3.0})
    ai.set_pricing("groq", "mock-model", {"inputPer1M": 0.5, "outputPer1M": 1.0})
    
    await example_simple_chat(ai)
    await example_streaming(ai)
    await example_circuit_breaker()
    await example_streaming_cache()
    await example_analytics_and_health(ai)

if __name__ == "__main__":
    asyncio.run(main())

import time
from typing import Dict, List, Any, Optional
from ..types import ProviderStats, AnalyticsSnapshot

LATENCY_WINDOW = 50

class AnalyticsTracker:
    def __init__(self):
        self.buckets: Dict[str, dict] = {}
        self.since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def record(self, provider: str, latency: int, tokens: int, cost: float, success: bool) -> None:
        if provider not in self.buckets:
            self.buckets[provider] = {
                "requests": 0,
                "errors": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "total_latency": 0,
                "recent_latencies": [],
            }
        
        b = self.buckets[provider]
        b["requests"] += 1
        b["total_tokens"] += tokens
        b["total_cost"] += cost

        if not success:
            b["errors"] += 1
        else:
            b["total_latency"] += latency
            b["recent_latencies"].append(latency)
            if len(b["recent_latencies"]) > LATENCY_WINDOW:
                b["recent_latencies"].pop(0)

    def get_provider_stats(self, provider: str) -> Optional[ProviderStats]:
        b = self.buckets.get(provider)
        if not b:
            return None
        return self._to_stats(provider, b)

    def get_snapshot(self) -> AnalyticsSnapshot:
        providers = []
        total_requests = 0
        total_errors = 0
        total_tokens = 0
        total_cost = 0.0

        for name, b in self.buckets.items():
            stats = self._to_stats(name, b)
            providers.append(stats)
            total_requests += stats.requests
            total_errors += stats.errors
            total_tokens += stats.total_tokens
            total_cost += stats.total_cost

        return AnalyticsSnapshot(
            total_requests=total_requests,
            total_errors=total_errors,
            total_tokens=total_tokens,
            total_cost=round(total_cost, 6),
            providers=providers,
            since=self.since,
        )

    def reset(self) -> None:
        self.buckets.clear()

    def _to_stats(self, provider: str, b: dict) -> ProviderStats:
        avg_latency = 0.0
        if b["recent_latencies"]:
            avg_latency = sum(b["recent_latencies"]) / len(b["recent_latencies"])

        return ProviderStats(
            provider=provider,
            requests=b["requests"],
            errors=b["errors"],
            total_tokens=b["total_tokens"],
            total_cost=round(b["total_cost"], 6),
            total_latency=b["total_latency"],
            avg_latency=int(round(avg_latency)),
            error_rate=b["errors"] / b["requests"] if b["requests"] > 0 else 0.0,
        )

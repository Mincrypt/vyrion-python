import time
from typing import Dict, Any, Optional

class CircuitBreakerManager:
    def __init__(self, failures_threshold: int = 3, cooldown_seconds: int = 60):
        self.failures_threshold = failures_threshold
        self.cooldown_seconds = cooldown_seconds
        self._failures: Dict[str, int] = {}
        self._cooldowns: Dict[str, float] = {}

    def record_success(self, provider: str) -> None:
        self._failures[provider] = 0

    def record_failure(self, provider: str, error: Any) -> None:
        is_rate_limit = self._is_rate_limit(error)
        count = self._failures.get(provider, 0) + 1
        self._failures[provider] = count

        if is_rate_limit or count >= self.failures_threshold:
            cooldown_until = time.time() + self.cooldown_seconds
            self._cooldowns[provider] = cooldown_until
            self._failures[provider] = 0

    def is_available(self, provider: str) -> bool:
        until = self._cooldowns.get(provider)
        if not until:
            return True
        if time.time() >= until:
            del self._cooldowns[provider]
            return True
        return False

    def get_cooldown_time_left(self, provider: str) -> float:
        until = self._cooldowns.get(provider)
        if not until:
            return 0.0
        left = until - time.time()
        return max(left, 0.0)

    def _is_rate_limit(self, error: Any) -> bool:
        if not error:
            return False
        msg = str(error).lower()
        if any(term in msg for term in ("429", "rate_limit", "rate limit", "too many requests")):
            return True
        for attr in ("status", "status_code", "statusCode"):
            if hasattr(error, attr) and getattr(error, attr) == 429:
                return True
        # Try checking dict representation if any (like response attributes)
        if isinstance(error, dict) and error.get("status") == 429:
            return True
        return False

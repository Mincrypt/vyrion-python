import asyncio
import time
from typing import Dict, List, Any, Optional
from ..types import HealthCheckResult

class HealthMonitor:
    def __init__(self, interval_seconds: int = 300):
        self.interval_seconds = interval_seconds
        self.cache: Dict[str, HealthCheckResult] = {}
        self.providers: list = []
        self._task: Optional[asyncio.Task] = None

    async def _loop(self):
        try:
            # Sleep first because start() calls check_all immediately
            while True:
                await asyncio.sleep(self.interval_seconds)
                await self.check_all()
        except asyncio.CancelledError:
            pass

    def start(self, providers: list) -> None:
        self.providers = providers
        # Run immediately in background
        asyncio.create_task(self.check_all())
        self.stop()
        self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None

    def get_status(self, provider: str) -> str:
        res = self.cache.get(provider)
        if not res:
            return "unknown"
        return res.status

    def get_all_statuses(self) -> List[HealthCheckResult]:
        return list(self.cache.values())

    async def check_all(self) -> List[HealthCheckResult]:
        tasks = []
        available_providers = [p for p in self.providers if p.is_available()]
        
        for p in available_providers:
            tasks.append(self.check(p))
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        resolved = []
        for i, res in enumerate(results):
            p = available_providers[i]
            if isinstance(res, Exception):
                failed = HealthCheckResult(
                    provider=p.name,
                    status="down",
                    latency=0,
                    checkedAt=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    error=str(res)
                )
                self.cache[p.name] = failed
                resolved.append(failed)
            else:
                resolved.append(res)
        return resolved

    async def check(self, provider: Any) -> HealthCheckResult:
        try:
            res_dict = await provider.health_check()
            if isinstance(res_dict, dict):
                result = HealthCheckResult(
                    provider=res_dict.get("provider", provider.name),
                    status=res_dict.get("status", "down"),
                    latency=res_dict.get("latency", 0),
                    checkedAt=res_dict.get("checkedAt", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
                    error=res_dict.get("error")
                )
            else:
                result = res_dict
            self.cache[provider.name] = result
            return result
        except Exception as err:
            failed = HealthCheckResult(
                provider=provider.name,
                status="down",
                latency=0,
                checkedAt=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                error=str(err)
            )
            self.cache[provider.name] = failed
            return failed

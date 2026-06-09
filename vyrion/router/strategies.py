from typing import List, Any, Callable

DEFAULT_CHEAPEST = ["ollama", "groq", "together", "gemini", "mistral", "anthropic", "openai"]
DEFAULT_BEST = ["openai", "anthropic", "gemini", "mistral", "groq", "together", "ollama"]

def pick_cheapest(providers: List[Any], analytics: Any) -> Any:
    order_dict = {name: i for i, name in enumerate(DEFAULT_CHEAPEST)}
    return min(providers, key=lambda p: order_dict.get(p.name, 999))

def pick_best(providers: List[Any], analytics: Any) -> Any:
    order_dict = {name: i for i, name in enumerate(DEFAULT_BEST)}
    return min(providers, key=lambda p: order_dict.get(p.name, 999))

def pick_fastest(providers: List[Any], analytics: Any) -> Any:
    if analytics and hasattr(analytics, "get_avg_latency"):
        return min(providers, key=lambda p: analytics.get_avg_latency(p.name))
    return pick_best(providers, analytics)

def resolve_strategy(goal: str) -> Callable[[List[Any], Any], Any]:
    if goal == "cheapest":
        return pick_cheapest
    elif goal == "fastest":
        return pick_fastest
    elif goal == "best":
        return pick_best
    return pick_best

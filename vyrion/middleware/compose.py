from typing import List, Callable, Any, Coroutine

def compose(middlewares: List[Callable[[Any, Callable[[], Coroutine[Any, Any, Any]]], Coroutine[Any, Any, Any]]]):
    def composed(context: Any, next_fn: Callable[[], Coroutine[Any, Any, Any]]) -> Coroutine[Any, Any, Any]:
        last_index = -1

        async def dispatch(i: int) -> Any:
            nonlocal last_index
            if i <= last_index:
                raise RuntimeError("next() called multiple times")
            last_index = i

            fn = middlewares[i] if i < len(middlewares) else next_fn
            if fn is None:
                return None

            async def next_dispatch():
                return await dispatch(i + 1)

            return await fn(context, next_dispatch)

        return dispatch(0)

    return composed

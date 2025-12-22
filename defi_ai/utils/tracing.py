from __future__ import annotations

from typing import Any, Callable, Optional


def traceable(
    *,
    name: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Best-effort trace decorator.

    If `langsmith` is installed, delegates to `langsmith.traceable`.
    Otherwise returns a no-op decorator (identity), keeping the repo runnable
    without LangSmith dependencies.
    """
    try:
        from langsmith import traceable as _ls_traceable  # type: ignore
        return _ls_traceable(name=name, tags=tags, metadata=metadata)
    except Exception:
        def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            return fn
        return _decorator

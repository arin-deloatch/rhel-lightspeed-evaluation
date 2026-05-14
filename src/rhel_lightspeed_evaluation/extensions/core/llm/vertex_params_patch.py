"""Patch litellm completion wrappers to support per-model vertex_project/vertex_location.

litellm.drop_params=True (set by DeepEvalLLMManager) silently strips vertex_project
and vertex_location from completion kwargs because they aren't in the provider's
supported-params list. This patch intercepts those params before they reach litellm
and temporarily sets them as module-level attributes, which litellm checks as a
fallback in its vertex_ai handler.

Usage in YAML config:

    judge_vertex_llama:
      provider: vertex_ai
      model: meta/llama-3.3-70b-instruct-maas
      parameters:
        max_completion_tokens: 2048
        vertex_location: us-central1   # per-model region

Applied explicitly from run_evaluation(), after deepeval_watsonx_patch.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import wraps
from typing import Any

import litellm
from lightspeed_evaluation.core.llm.litellm_patch import (
    _original_acompletion,
    _original_completion,
    litellm_state_lock,
)
from lightspeed_evaluation.core.llm.token_tracker import track_tokens

logger = logging.getLogger(__name__)


@contextmanager
def _vertex_override(kwargs: dict[str, Any]):
    """Pop vertex_project/vertex_location from kwargs and set as litellm module attrs.

    When neither key is present the context manager is a no-op (no lock acquired).
    """
    vp = kwargs.pop("vertex_project", None)
    vl = kwargs.pop("vertex_location", None)
    if vp is None and vl is None:
        yield
        return
    with litellm_state_lock:
        old_vp = getattr(litellm, "vertex_project", None)
        old_vl = getattr(litellm, "vertex_location", None)
        try:
            if vp is not None:
                litellm.vertex_project = vp
            if vl is not None:
                litellm.vertex_location = vl
            yield
        finally:
            litellm.vertex_project = old_vp
            litellm.vertex_location = old_vl


@wraps(_original_completion)
def _completion_with_vertex_and_tracking(*args: Any, **kwargs: Any) -> Any:
    """Wrapper that handles vertex params, then delegates to the real litellm.completion."""
    with _vertex_override(kwargs):
        response = _original_completion(*args, **kwargs)
    try:
        track_tokens(response)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception("Failed to track tokens for completion: %s", e)
    return response


@wraps(_original_acompletion)
async def _acompletion_with_vertex_and_tracking(*args: Any, **kwargs: Any) -> Any:
    """Async wrapper that handles vertex params, then delegates to the real litellm.acompletion."""
    with _vertex_override(kwargs):
        response = await _original_acompletion(*args, **kwargs)
    try:
        track_tokens(response)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception("Failed to track tokens for acompletion: %s", e)
    return response


def apply_vertex_params_patch() -> None:
    """Replace litellm.completion/acompletion with vertex-aware wrappers."""
    litellm.completion = _completion_with_vertex_and_tracking
    litellm.acompletion = _acompletion_with_vertex_and_tracking
    logger.info("Applied vertex params patch for per-model project/location support")


__all__ = ["apply_vertex_params_patch"]

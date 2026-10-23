"""Patch DeepEval LiteLLM integration so WatsonX judges use WatsonX credentials.

DeepEval's ``LiteLLMModel`` selects ``api_key`` from LITELLM_* / OPENAI_* /
ANTHROPIC_* / GOOGLE_* settings only. If ``OPENAI_API_KEY`` is set (common when
the system LLM is OpenAI), that key is injected into every ``acompletion`` call.

For ``watsonx/...`` models, LiteLLM treats the explicit ``api_key`` as the IBM
API key for IAM token exchange, so IBM returns BXNIM0415E ("key not found").

This module replaces ``DeepEvalLLMManager`` so that WatsonX judges pass the
``WX_API_KEY`` / ``WATSONX_API_KEY`` / ``WATSONX_APIKEY`` value explicitly.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from lightspeed_evaluation.core.llm.deepeval import DeepEvalLLMManager as _BaseDeepEvalLLMManager

logger = logging.getLogger(__name__)


def _watsonx_api_key_from_env() -> str | None:
    return os.getenv("WX_API_KEY") or os.getenv("WATSONX_API_KEY") or os.getenv("WATSONX_APIKEY")


class DeepEvalLLMManager(_BaseDeepEvalLLMManager):
    """Same as upstream, but WatsonX models get credentials from WatsonX env vars."""

    def __init__(self, model_name: str, llm_params: dict[str, Any]) -> None:
        # Delegate retry/SSL setup to upstream; only override WatsonX credentials.
        params = {**llm_params, "parameters": dict(llm_params.get("parameters", {}))}
        if model_name.startswith("watsonx/"):
            wx_key = _watsonx_api_key_from_env()
            if wx_key:
                params["parameters"]["api_key"] = wx_key
        super().__init__(model_name, params)


def apply_deepeval_watsonx_patch() -> None:
    """Monkey-patch lightspeed_evaluation to use WatsonX-aware DeepEvalLLMManager."""
    import lightspeed_evaluation.core.llm.deepeval as deepeval_mod

    deepeval_mod.DeepEvalLLMManager = DeepEvalLLMManager


__all__ = ["DeepEvalLLMManager", "apply_deepeval_watsonx_patch"]

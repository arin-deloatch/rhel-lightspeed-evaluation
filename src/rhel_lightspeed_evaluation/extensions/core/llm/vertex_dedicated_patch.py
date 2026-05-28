"""Support for self-deployed Vertex AI Model Garden endpoints (vertex_dedicated provider).

Resolves dedicated endpoint DNS at startup, patches LLMManager to handle the
vertex_dedicated provider, and injects per-call GCP token refresh for dedicated
endpoint calls.

Usage in YAML config:

    judge_granite:
      provider: vertex_dedicated
      model: ibm-granite/granite-4.1-8b
      parameters:
        endpoint_id: mg-endpoint-cdd2a610-ffe9-48d8-8370-9ab335a77f7a
        project: rhel-lightspeed-650189
        region: us-central1

Applied from run_evaluation(), after vertex_params_patch.
"""

from __future__ import annotations

import logging
import os
import subprocess
from functools import wraps
from typing import TYPE_CHECKING, Any

from lightspeed_evaluation.core.llm.manager import LLMManager
from lightspeed_evaluation.core.system.exceptions import ConfigurationError, LLMError

if TYPE_CHECKING:
    from lightspeed_evaluation.core.models import LLMConfig

logger = logging.getLogger(__name__)

_gcp_credentials: Any = None


def _get_gcp_credentials() -> Any:
    """Return cached GCP default credentials, loading on first call."""
    global _gcp_credentials  # noqa: PLW0603
    if _gcp_credentials is None:
        import google.auth

        _gcp_credentials, _ = google.auth.default()
    return _gcp_credentials


def _resolve_dedicated_dns(endpoint_id: str, project: str, region: str) -> str:
    """Resolve dedicated endpoint DNS via the gcloud CLI."""
    try:
        result = subprocess.run(
            [
                "gcloud",
                "ai",
                "endpoints",
                "describe",
                endpoint_id,
                f"--region={region}",
                f"--project={project}",
                "--format=value(dedicatedEndpointDns)",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        dns = result.stdout.strip()
        if not dns:
            stderr = result.stderr.strip()
            raise ConfigurationError(
                f"Could not resolve dedicated DNS for endpoint {endpoint_id}: {stderr}"
            )
        return dns
    except FileNotFoundError:
        raise ConfigurationError(
            "gcloud CLI not found. "
            "Install the Google Cloud SDK to use the vertex_dedicated provider."
        ) from None
    except subprocess.TimeoutExpired as exc:
        raise ConfigurationError(
            f"Timeout resolving dedicated DNS for endpoint {endpoint_id}"
        ) from exc


def _handle_vertex_dedicated(manager: LLMManager) -> str:
    """Provider handler for vertex_dedicated.

    Resolves the dedicated endpoint DNS, constructs base_url, and returns the
    model name with an ``openai/`` prefix so litellm routes via the
    OpenAI-compatible path.
    """
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise LLMError(
            "GOOGLE_APPLICATION_CREDENTIALS environment variable "
            "is required for the vertex_dedicated provider"
        )

    params: dict[str, Any] = manager.config.parameters
    endpoint_id = params.pop("endpoint_id", None)
    project = params.pop("project", None)
    region = params.pop("region", None)

    if not all([endpoint_id, project, region]):
        raise ConfigurationError(
            "vertex_dedicated provider requires endpoint_id, project, and region "
            "in parameters"
        )

    dns = _resolve_dedicated_dns(endpoint_id, project, region)
    base_url = (
        f"https://{dns}/v1/projects/{project}"
        f"/locations/{region}/endpoints/{endpoint_id}"
    )
    params["base_url"] = base_url
    logger.info("Resolved vertex_dedicated endpoint: %s", base_url)

    return f"openai/{manager.config.model}"


def _apply_llm_manager_patch() -> None:
    """Extend LLMManager to recognise the vertex_dedicated provider."""
    _original_construct = LLMManager._construct_model_name_and_validate

    def _construct_with_vertex_dedicated(self: LLMManager, config: LLMConfig) -> str:
        if config.provider.lower() == "vertex_dedicated":
            return _handle_vertex_dedicated(self)
        return _original_construct(self, config)

    LLMManager._construct_model_name_and_validate = (  # type: ignore[assignment]
        _construct_with_vertex_dedicated
    )


def _apply_litellm_gcp_refresh() -> None:
    """Wrap litellm completion functions to refresh GCP tokens for dedicated endpoints."""
    import lightspeed_evaluation.core.llm.litellm_patch as lp

    _real_completion = lp._original_completion
    _real_acompletion = lp._original_acompletion

    @wraps(_real_completion)
    def _completion_with_gcp_refresh(*args: Any, **kwargs: Any) -> Any:
        if "prediction.vertexai.goog" in str(kwargs.get("api_base") or ""):
            import google.auth.transport.requests

            creds = _get_gcp_credentials()
            creds.refresh(google.auth.transport.requests.Request())
            kwargs["api_key"] = creds.token
            kwargs.setdefault("extra_body", {})["@requestFormat"] = "chatCompletions"
            logger.debug("Refreshed GCP token for dedicated endpoint call")
        return _real_completion(*args, **kwargs)

    @wraps(_real_acompletion)
    async def _acompletion_with_gcp_refresh(*args: Any, **kwargs: Any) -> Any:
        if "prediction.vertexai.goog" in str(kwargs.get("api_base") or ""):
            import google.auth.transport.requests

            creds = _get_gcp_credentials()
            creds.refresh(google.auth.transport.requests.Request())
            kwargs["api_key"] = creds.token
            kwargs.setdefault("extra_body", {})["@requestFormat"] = "chatCompletions"
            logger.debug("Refreshed GCP token for dedicated endpoint call")
        return await _real_acompletion(*args, **kwargs)

    lp._original_completion = _completion_with_gcp_refresh
    lp._original_acompletion = _acompletion_with_gcp_refresh

    try:
        import rhel_lightspeed_evaluation.extensions.core.llm.vertex_params_patch as vp

        vp._original_completion = _completion_with_gcp_refresh  # type: ignore[attr-defined]
        vp._original_acompletion = _acompletion_with_gcp_refresh  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        pass


def apply_vertex_dedicated_patch() -> None:
    """Add vertex_dedicated provider support to the framework."""
    _apply_llm_manager_patch()
    _apply_litellm_gcp_refresh()
    logger.info("Applied vertex_dedicated patch for self-deployed Vertex AI endpoints")


__all__ = ["apply_vertex_dedicated_patch"]

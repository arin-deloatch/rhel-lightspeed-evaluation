"""Extended system configuration models for RHEL Lightspeed Evaluation.

Adds support for 'chat/completions' API endpoint type.
"""

from lightspeed_evaluation.core.models import APIConfig
from lightspeed_evaluation.core.models.agents import AgentsConfig, HttpApiAgentConfig
from lightspeed_evaluation.core.models.system import SystemConfig
from pydantic import Field, field_validator

EXTENDED_ENDPOINT_TYPES = {"query", "streaming", "chat/completions", "infer"}


def _validate_extended_endpoint_type(v: str) -> str:
    if not isinstance(v, str):
        raise ValueError("endpoint_type must be a string")
    v = v.strip()
    if v not in EXTENDED_ENDPOINT_TYPES:
        raise ValueError(
            f"Unsupported endpoint_type: '{v}'. Supported types: {sorted(EXTENDED_ENDPOINT_TYPES)}"
        )
    return v


class APIConfigExt(APIConfig):
    """Extended API configuration that supports 'chat/completions' endpoint type."""

    endpoint_type: str = "streaming"

    @field_validator("endpoint_type", mode="before")
    @classmethod
    def validate_endpoint_type(cls, v: str) -> str:
        """Validate endpoint_type is one of the supported values."""
        return _validate_extended_endpoint_type(v)


class HttpApiAgentConfigExt(HttpApiAgentConfig):
    """Extended HTTP API agent config that supports 'chat/completions' endpoint type."""

    @field_validator("endpoint_type", mode="before")
    @classmethod
    def validate_endpoint_type(cls, v: str) -> str:
        """Validate endpoint_type is one of the supported values."""
        return _validate_extended_endpoint_type(v)


class AgentsConfigExt(AgentsConfig):
    """Extended AgentsConfig that uses HttpApiAgentConfigExt."""

    agents: dict[str, HttpApiAgentConfigExt] = Field(default_factory=dict)


class SystemConfigExt(SystemConfig):
    """Extended SystemConfig with chat/completions API support."""

    api: APIConfigExt = Field(
        default_factory=lambda: APIConfigExt(**{}),
        description="API configuration (extended to support chat/completions endpoint)",
    )
    agents: AgentsConfigExt | None = Field(
        default=None,
        description="Agents configuration (extended to support chat/completions endpoint)",
    )

"""Extended system configuration models for RHEL Lightspeed Evaluation.

Adds support for 'chat/completions' API endpoint type.
"""

from lightspeed_evaluation.core.models import APIConfig
from lightspeed_evaluation.core.models.system import SystemConfig
from pydantic import Field, field_validator


class APIConfigExt(APIConfig):
    """Extended API configuration that supports 'chat/completions' endpoint type."""

    endpoint_type: str = "streaming"

    @field_validator("endpoint_type", mode="before")
    @classmethod
    def validate_endpoint_type(cls, v: str) -> str:
        """Validate endpoint_type is one of the supported values."""
        if not isinstance(v, str):
            raise ValueError("endpoint_type must be a string")

        v = v.strip()
        allowed_endpoints = {"query", "streaming", "chat/completions", "infer"}

        if v not in allowed_endpoints:
            raise ValueError(
                f"Unsupported endpoint_type: '{v}'. Supported types: {sorted(allowed_endpoints)}"
            )

        return v


class SystemConfigExt(SystemConfig):
    """Extended SystemConfig with chat/completions API support."""

    api: APIConfigExt = Field(
        default_factory=lambda: APIConfigExt(**{}),
        description="API configuration (extended to support chat/completions endpoint)",
    )

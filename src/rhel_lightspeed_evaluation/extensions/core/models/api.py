from typing import Any

from lightspeed_evaluation.core.models import APIResponse, AttachmentData
from pydantic import BaseModel, ConfigDict, Field


class APIResponseExt(APIResponse):
    """Extended API response that tolerates null tool_calls."""

    model_config = ConfigDict(extra="forbid")

    tool_calls: list[list[dict[str, Any]]] | None = Field(
        default_factory=list, description="Tool call sequences"
    )

    @classmethod
    def from_raw_response(cls, raw_data: dict[str, Any]) -> "APIResponseExt":
        """Create APIResponseExt from raw API response data, normalizing null tool_calls."""
        if raw_data.get("tool_calls") is None:
            raw_data["tool_calls"] = []
        return super().from_raw_response(raw_data)


class APIRequestExt(BaseModel):
    """API request model for dynamic data generation."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="User query")
    messages: list[dict[str, str]] | None = Field(
        default=None, description="User query formatted as messages"
    )
    provider: str | None = Field(default=None, description="LLM provider")
    model: str | None = Field(default=None, description="LLM model")
    no_tools: bool | None = Field(default=None, description="Disable tool usage")
    conversation_id: str | None = Field(
        default=None, description="Conversation ID for context"
    )
    system_prompt: str | None = Field(
        default=None, description="System prompt override"
    )
    attachments: list[AttachmentData] | None = Field(
        default=None, description="File attachments"
    )

    @classmethod
    def create(
        cls,
        query: str,
        **kwargs: Any,
    ) -> "APIRequestExt":
        """Create API request with optional attachments."""
        # Extract parameters with defaults
        provider = kwargs.get("provider")
        model = kwargs.get("model")
        no_tools = kwargs.get("no_tools")
        conversation_id = kwargs.get("conversation_id")
        system_prompt = kwargs.get("system_prompt")
        attachments = kwargs.get("attachments")
        attachment_data = None
        if attachments:
            attachment_data = [
                AttachmentData(content=attachment) for attachment in attachments
            ]

        return cls(
            query=query,
            messages=[{"role": "user", "content": query}],
            provider=provider,
            model=model,
            no_tools=no_tools,
            conversation_id=conversation_id,
            system_prompt=system_prompt,
            attachments=attachment_data,
        )


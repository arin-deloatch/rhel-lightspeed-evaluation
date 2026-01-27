from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


from lightspeed_evaluation.core.models import AttachmentData

class APIRequestExt(BaseModel):
    """API request model for dynamic data generation."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="User query")
    messages: Optional[list[dict[str, str]]] = Field(
        default=None, description="User query formatted as messages"
    )
    provider: Optional[str] = Field(default=None, description="LLM provider")
    model: Optional[str] = Field(default=None, description="LLM model")
    no_tools: Optional[bool] = Field(default=None, description="Disable tool usage")
    conversation_id: Optional[str] = Field(
        default=None, description="Conversation ID for context"
    )
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt override"
    )
    attachments: Optional[list[AttachmentData]] = Field(
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


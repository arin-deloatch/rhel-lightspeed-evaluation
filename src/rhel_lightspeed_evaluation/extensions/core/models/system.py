"""System configuration models."""
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lightspeed_evaluation.core.constants import (
    DEFAULT_BASE_FILENAME,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_OUTPUT_DIR,
    SUPPORTED_OUTPUT_TYPES
)

from rhel_lightspeed_evaluation.extensions.core.constants import SUPPORTED_CSV_COLUMNS


from lightspeed_evaluation.core.models.system import SystemConfig
from pydantic import BaseModel, ConfigDict, Field, field_validator

class OutputConfig(BaseModel):
    """Output configuration for evaluation results."""

    model_config = ConfigDict(extra="forbid")

    output_dir: str = Field(
        default=DEFAULT_OUTPUT_DIR, description="Output directory for results"
    )
    base_filename: str = Field(
        default=DEFAULT_BASE_FILENAME, description="Base filename for output files"
    )
    enabled_outputs: list[str] = Field(
        default=SUPPORTED_OUTPUT_TYPES,
        description="List of enabled output types: csv, json, txt",
    )
    csv_columns: list[str] = Field(
        default=SUPPORTED_CSV_COLUMNS,
        description="CSV columns to include in detailed results",
    )
    summary_config_sections: list[str] = Field(
        default=["llm", "embedding", "api", "panel_of_judges"],
        description="Configuration sections to include in summary reports",
    )

    @field_validator("csv_columns")
    @classmethod
    def validate_csv_columns(cls, v: list[str]) -> list[str]:
        """Validate that all CSV columns are supported."""
        for column in v:
            if column not in SUPPORTED_CSV_COLUMNS:
                raise ValueError(
                    f"Unsupported CSV column: {column}. "
                    f"Supported columns: {SUPPORTED_CSV_COLUMNS}"
                )
        return v

    @field_validator("enabled_outputs")
    @classmethod
    def validate_enabled_outputs(cls, v: list[str]) -> list[str]:
        """Validate that all enabled outputs are supported."""
        for output_type in v:
            if output_type not in SUPPORTED_OUTPUT_TYPES:
                raise ValueError(
                    f"Unsupported output type: {output_type}. "
                    f"Supported types: {SUPPORTED_OUTPUT_TYPES}"
                )
        return v


class JudgeConfig(BaseModel):
    """Configuration for a single judge in the panel."""

    model_config = ConfigDict(extra="forbid")

    judge_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for this judge (auto-generated if not provided)",
    )
    provider: str = Field(
        min_length=1,
        description="LLM provider for this judge (e.g., openai, anthropic, gemini)",
    )
    model: str = Field(
        min_length=1,
        description="Model identifier for this judge",
    )
    temperature: float = Field(
        default=DEFAULT_LLM_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for this judge",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum tokens for this judge (uses system default if not specified)",
    )
    timeout: Optional[int] = Field(
        default=None,
        ge=1,
        description="Request timeout for this judge (uses system default if not specified)",
    )
    num_retries: Optional[int] = Field(
        default=None,
        ge=0,
        description="Retry attempts for this judge (uses system default if not specified)",
    )


class PanelOfJudgesConfig(BaseModel):
    """Configuration for panel of judges evaluation."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Enable panel of judges evaluation (uses single LLM if false)",
    )
    apply_to: list[str] = Field(
        default_factory=lambda: ["geval", "custom"],
        description="Metric types that should use panel evaluation (e.g., 'geval', 'custom')",
    )
    aggregation_method: str = Field(
        default="mean",
        description="Method to aggregate scores from judges (mean, median, weighted_mean)",
    )
    output_individual_scores: bool = Field(
        default=False,
        description="Include individual judge scores in CSV output (verbose mode)",
    )
    judges: list[JudgeConfig] = Field(
        default_factory=list,
        min_length=1,
        description="List of judge configurations (at least one judge required if enabled)",
    )

    @field_validator("aggregation_method")
    @classmethod
    def validate_aggregation_method(cls, v: str) -> str:
        """Validate aggregation method is supported."""
        allowed_methods = {"mean", "median", "weighted_mean"}
        if v not in allowed_methods:
            raise ValueError(
                f"Unsupported aggregation method '{v}'. Allowed: {sorted(allowed_methods)}"
            )
        return v

    @field_validator("apply_to")
    @classmethod
    def validate_apply_to(cls, v: list[str]) -> list[str]:
        """Validate metric types are supported."""
        allowed_types = {"geval", "custom", "deepeval"}
        for metric_type in v:
            if metric_type not in allowed_types:
                raise ValueError(
                    f"Unsupported metric type '{metric_type}'. Allowed: {sorted(allowed_types)}"
                )
        return v

    @field_validator("judges")
    @classmethod
    def validate_judges(cls, v: list[JudgeConfig]) -> list[JudgeConfig]:
        """Validate judges configuration and auto-generate IDs if needed."""
        # Auto-generate judge IDs if not provided
        # Format: provider_model (e.g., "openai_gpt-4", "anthropic_claude-3-sonnet")
        for judge in v:
            if not judge.judge_id:
                # Create descriptive ID from provider and model
                # Sanitize model name to remove slashes and special chars
                sanitized_model = (
                    judge.model.replace("/", "_").replace(":", "_").replace(".", "_")
                )
                judge.judge_id = f"{judge.provider}_{sanitized_model}"

        # Check for duplicate judge IDs
        judge_ids = [judge.judge_id for judge in v]
        if len(judge_ids) != len(set(judge_ids)):
            # If duplicates exist (e.g., same model used twice), add numeric suffix
            seen: dict = {}
            for judge in v:
                if judge.judge_id in seen:
                    seen[judge.judge_id] += 1
                    judge.judge_id = f"{judge.judge_id}_{seen[judge.judge_id]}"
                else:
                    seen[judge.judge_id] = 1

        return v
class SystemConfigExt(SystemConfig):
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Output configuration (extended with csv_columns)",
    )
    panel_of_judges: PanelOfJudgesConfig = Field(
        default_factory=PanelOfJudgesConfig,
        description="Panel of judges configuration (optional)",
    )

if __name__ == '__main__':
    # Sanity check 
    print("Child fields:", SystemConfigExt.model_fields)
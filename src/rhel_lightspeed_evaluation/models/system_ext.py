"""System configuration models."""

from typing import Any, Literal

from lightspeed_evaluation.core.constants import (
    DEFAULT_API_TIMEOUT,
    DEFAULT_BASE_FILENAME,
    DEFAULT_EMBEDDING_CACHE_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_LLM_CACHE_DIR,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_RETRIES,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_PACKAGE_LEVEL,
    DEFAULT_LOG_SHOW_TIMESTAMPS,
    DEFAULT_LOG_SOURCE_LEVEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VISUALIZATION_DPI,
    DEFAULT_VISUALIZATION_FIGSIZE,
    SUPPORTED_CSV_COLUMNS,
    SUPPORTED_GRAPH_TYPES,
    SUPPORTED_OUTPUT_TYPES,
)
from lightspeed_evaluation.core.models import APIConfig
from pydantic import BaseModel, ConfigDict, Field, field_validator


class LLMConfig(BaseModel):
    """LLM configuration from system configuration."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(
        default=DEFAULT_LLM_PROVIDER,
        min_length=1,
        description="Provider name, e.g., openai, azure, watsonx etc..",
    )
    model: str = Field(
        default=DEFAULT_LLM_MODEL,
        min_length=1,
        description="Model identifier or deployment name",
    )
    temperature: float = Field(
        default=DEFAULT_LLM_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_tokens: int = Field(
        default=DEFAULT_LLM_MAX_TOKENS, ge=1, description="Maximum tokens in response"
    )
    timeout: int = Field(
        default=DEFAULT_API_TIMEOUT, ge=1, description="Request timeout in seconds"
    )
    num_retries: int = Field(
        default=DEFAULT_LLM_RETRIES,
        ge=0,
        description="Retry attempts for failed requests",
    )
    cache_dir: str = Field(
        default=DEFAULT_LLM_CACHE_DIR,
        min_length=1,
        description="Location of cached 'LLM as a judge' queries",
    )
    cache_enabled: bool = Field(
        default=True, description="Is caching of 'LLM as a judge' queries enabled?"
    )


class JudgeConfig(LLMConfig):
    """Configuration for an individual judge in a panel."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, description="Judge identifier")
    provider: str = Field(
        default=DEFAULT_LLM_PROVIDER,
        min_length=1,
        description="LLM provider (openai, watsonx, azure, etc.)",
    )
    model: str = Field(
        default=DEFAULT_LLM_MODEL, min_length=1, description="Model identifier or deployment name"
    )
    temperature: float = Field(
        default=DEFAULT_LLM_TEMPERATURE, ge=0.0, le=2.0, description="Sampling temperature"
    )
    # max_tokens, timeout, num_retries, cache_dir, cache_enabled inherited from LLMConfig


class PanelConfig(BaseModel):
    """Panel-of-judges configuration."""

    model_config = ConfigDict(extra="forbid")

    enable_panel: bool = Field(
        default=False,
        description="Enable evaluation via a panel of multiple LLM judges (off by default).",
    )
    aggregation_strategy: Literal["average", "weighted_average", "majority_vote"] = Field(
        default="average",
        description="How to aggregate per-judge scores into a single score.",
    )
    judge_weights: dict[str, float] | None = Field(
        default=None,
        description="Weights for 'weighted_average' aggregation (keys are judge names).",
    )
    judges: list[JudgeConfig] | None = Field(
        default=None,
        description="List of judge configurations. Required when enable_panel = true.",
    )

    @field_validator("judge_weights")
    @classmethod
    def _validate_weights(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        if v is None:
            return v
        if any(w < 0 for w in v.values()):
            raise ValueError("judge_weights must be non-negative.")
        return v


class EmbeddingConfig(BaseModel):
    """Embedding configuration."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(
        default=DEFAULT_EMBEDDING_PROVIDER,
        min_length=1,
        description="Provider name, e.g., huggingface, openai",
    )
    model: str = Field(
        default=DEFAULT_EMBEDDING_MODEL,
        min_length=1,
        description="Embedding model identifier",
    )
    provider_kwargs: dict[str, Any] | None = Field(
        default=None,
        description="Embedding provider arguments, e.g. model_kwargs: device:cpu",
    )
    cache_dir: str = Field(
        default=DEFAULT_EMBEDDING_CACHE_DIR,
        min_length=1,
        description="Location of cached embedding queries",
    )
    cache_enabled: bool = Field(
        default=True, description="Is caching of embedding queries enabled?"
    )

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, v: str) -> str:
        allowed = {"openai", "huggingface"}
        if v not in allowed:
            raise ValueError(f"Unsupported embedding provider '{v}'. Allowed: {sorted(allowed)}")
        return v


class OutputConfig(BaseModel):
    """Output configuration for evaluation results."""

    model_config = ConfigDict(extra="forbid")

    output_dir: str = Field(default=DEFAULT_OUTPUT_DIR, description="Output directory for results")
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

    @field_validator("csv_columns")
    @classmethod
    def validate_csv_columns(cls, v: list[str]) -> list[str]:
        """Validate that all CSV columns are supported."""
        for column in v:
            if column not in SUPPORTED_CSV_COLUMNS:
                raise ValueError(
                    f"Unsupported CSV column: {column}. Supported columns: {SUPPORTED_CSV_COLUMNS}"
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


class LoggingConfig(BaseModel):
    """Logging configuration."""

    model_config = ConfigDict(extra="forbid")

    source_level: str = Field(
        default=DEFAULT_LOG_SOURCE_LEVEL, description="Source code logging level"
    )
    package_level: str = Field(
        default=DEFAULT_LOG_PACKAGE_LEVEL, description="Package logging level"
    )
    log_format: str = Field(default=DEFAULT_LOG_FORMAT, description="Log message format")
    show_timestamps: bool = Field(
        default=DEFAULT_LOG_SHOW_TIMESTAMPS, description="Show timestamps in logs"
    )
    package_overrides: dict[str, str] = Field(
        default_factory=dict, description="Package-specific log level overrides"
    )


class VisualizationConfig(BaseModel):
    """Visualization configuration for graphs and charts."""

    model_config = ConfigDict(extra="forbid")

    figsize: list[int] = Field(
        default=DEFAULT_VISUALIZATION_FIGSIZE, description="Figure size [width, height]"
    )
    dpi: int = Field(default=DEFAULT_VISUALIZATION_DPI, ge=50, description="Resolution in DPI")
    enabled_graphs: list[str] = Field(
        default=[],
        description="List of graph types to generate",
    )

    @field_validator("enabled_graphs")
    @classmethod
    def validate_enabled_graphs(cls, v: list[str]) -> list[str]:
        """Validate that all enabled graphs are supported."""
        for graph_type in v:
            if graph_type not in SUPPORTED_GRAPH_TYPES:
                raise ValueError(
                    f"Unsupported graph type: {graph_type}. "
                    f"Supported types: {SUPPORTED_GRAPH_TYPES}"
                )
        return v


class CoreConfig(BaseModel):
    """Core evaluation configuration (e.g., concurrency limits)."""

    model_config = ConfigDict(extra="forbid")

    max_threads: int | None = Field(
        default=None,
        description="Maximum threads for multithreading eval",
        gt=0,
    )


class GEvalConfig(BaseModel):
    """GEval metrics configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        description="Enable GEval metrics evaluation",
    )
    registry_path: str = Field(
        default="config/geval_metrics.yaml",
        description="Path to GEval metrics registry YAML file",
    )
    default_turn_metrics: list[str] = Field(
        default_factory=list,
        description="Default turn-level GEval metrics to auto-apply (e.g., ['geval:technical_accuracy'])",
    )
    default_conversation_metrics: list[str] = Field(
        default_factory=list,
        description="Default conversation-level GEval metrics to auto-apply (e.g., ['geval:conversation_coherence'])",
    )


class SystemConfig(BaseModel):
    """System configuration using individual config models."""

    model_config = ConfigDict(extra="forbid")

    # Individual configuration models
    core: CoreConfig = Field(default_factory=CoreConfig, description="Core eval configuration")
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration")
    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig, description="Embeddings configuration"
    )
    api: APIConfig = Field(default_factory=APIConfig, description="API configuration")
    output: OutputConfig = Field(default_factory=OutputConfig, description="Output configuration")
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration"
    )
    visualization: VisualizationConfig = Field(
        default_factory=VisualizationConfig, description="Visualization configuration"
    )

    panel: PanelConfig = Field(
        default_factory=PanelConfig,
        description="Configuration for panel-of-judges feature (disabled by default).",
    )
    geval: GEvalConfig = Field(
        default_factory=GEvalConfig,
        description="GEval metrics configuration",
    )

    # Default metrics metadata from system config
    default_turn_metrics_metadata: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Default turn metrics metadata"
    )
    default_conversation_metrics_metadata: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Default conversation metrics metadata"
    )

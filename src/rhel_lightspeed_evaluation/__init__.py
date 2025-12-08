"""RHEL Lightspeed Evaluation Extensions

This package provides extensions to the lightspeed-evaluation framework,
including:
- Panel of Judges for multi-model evaluation
- Config extensions to support panel of judges.

Main components:
- EvaluationPipelineExt: Extended evaluation pipeline with custom processing
- Extended metrics: DeepEval, GEval, and custom metric implementations
- Extended models: SystemConfigExt, EvaluationResultExt
- Extended processors: ConversationProcessorExt, MetricsEvaluatorExt
- Extended output handlers: OutputHandlerExt

This package also re-exports key components from the upstream lightspeed_evaluation
framework for convenience.
"""

from typing import TYPE_CHECKING

from lightspeed_evaluation.core.system.lazy_import import create_lazy_getattr

if TYPE_CHECKING:
    # ruff: noqa: F401
    # Extended pipeline components
    from rhel_lightspeed_evaluation.extensions.pipeline.evaluation import (
        ConversationProcessorExt,
        EvaluationPipelineExt,
        MetricsEvaluatorExt,
    )

    # Extended core components
    from rhel_lightspeed_evaluation.extensions.core.llm import DeepEvalLLMManagerExt
    from rhel_lightspeed_evaluation.extensions.core.metrics import (
        CustomMetricsExt,
        DeepEvalMetricsExt,
        GEvalHandlerExt,
    )
    from rhel_lightspeed_evaluation.extensions.core.models import (
        EvaluationResultExt,
        SystemConfigExt,
    )
    from rhel_lightspeed_evaluation.extensions.core.output import OutputHandlerExt
    from rhel_lightspeed_evaluation.extensions.core.system import ConfigLoaderExt

    # Re-export key upstream components for convenience
    from lightspeed_evaluation.core.api import APIClient
    from lightspeed_evaluation.core.llm import LLMManager
    from lightspeed_evaluation.core.models import (
        APIConfig,
        EvaluationData,
        EvaluationResult,
        LLMConfig,
        LoggingConfig,
        OutputConfig,
        TurnData,
        VisualizationConfig,
    )
    from lightspeed_evaluation.core.output import GraphGenerator, OutputHandler
    from lightspeed_evaluation.core.script import ScriptExecutionManager
    from lightspeed_evaluation.core.system import (
        ConfigLoader,
        DataValidator,
        SystemConfig,
    )
    from lightspeed_evaluation.core.system.exceptions import (
        APIError,
        DataValidationError,
        EvaluationError,
        ScriptExecutionError,
    )
    from lightspeed_evaluation.pipeline.evaluation import EvaluationPipeline

__version__ = "0.1.0"

__author__ = "Arin DeLoatch"

_LAZY_IMPORTS = {
    # Extended pipeline
    "EvaluationPipelineExt": (
        "rhel_lightspeed_evaluation.extensions.pipeline.evaluation.pipeline",
        "EvaluationPipelineExt",
    ),
    "ConversationProcessorExt": (
        "rhel_lightspeed_evaluation.extensions.pipeline.evaluation",
        "ConversationProcessorExt",
    ),
    "MetricsEvaluatorExt": (
        "rhel_lightspeed_evaluation.extensions.pipeline.evaluation",
        "MetricsEvaluatorExt",
    ),
    # Extended system config
    "ConfigLoaderExt": (
        "rhel_lightspeed_evaluation.extensions.core.system",
        "ConfigLoaderExt",
    ),
    "SystemConfigExt": (
        "rhel_lightspeed_evaluation.extensions.core.models",
        "SystemConfigExt",
    ),
    # Extended models
    "EvaluationResultExt": (
        "rhel_lightspeed_evaluation.extensions.core.models",
        "EvaluationResultExt",
    ),
    # Extended metrics
    "DeepEvalMetricsExt": (
        "rhel_lightspeed_evaluation.extensions.core.metrics",
        "DeepEvalMetricsExt",
    ),
    "CustomMetricsExt": (
        "rhel_lightspeed_evaluation.extensions.core.metrics",
        "CustomMetricsExt",
    ),
    "GEvalHandlerExt": (
        "rhel_lightspeed_evaluation.extensions.core.metrics",
        "GEvalHandlerExt",
    ),
    # Extended LLM
    "DeepEvalLLMManagerExt": (
        "rhel_lightspeed_evaluation.extensions.core.llm",
        "DeepEvalLLMManagerExt",
    ),
    # Extended output handling
    "OutputHandlerExt": (
        "rhel_lightspeed_evaluation.extensions.core.output",
        "OutputHandlerExt",
    ),
    # Re-export upstream components
    "EvaluationPipeline": (
        "lightspeed_evaluation.pipeline.evaluation",
        "EvaluationPipeline",
    ),
    "ConfigLoader": ("lightspeed_evaluation.core.system", "ConfigLoader"),
    "SystemConfig": ("lightspeed_evaluation.core.system", "SystemConfig"),
    "DataValidator": ("lightspeed_evaluation.core.system", "DataValidator"),
    "LLMConfig": ("lightspeed_evaluation.core.models", "LLMConfig"),
    "APIConfig": ("lightspeed_evaluation.core.models", "APIConfig"),
    "OutputConfig": ("lightspeed_evaluation.core.models", "OutputConfig"),
    "LoggingConfig": ("lightspeed_evaluation.core.models", "LoggingConfig"),
    "VisualizationConfig": ("lightspeed_evaluation.core.models", "VisualizationConfig"),
    "EvaluationData": ("lightspeed_evaluation.core.models", "EvaluationData"),
    "TurnData": ("lightspeed_evaluation.core.models", "TurnData"),
    "EvaluationResult": ("lightspeed_evaluation.core.models", "EvaluationResult"),
    "LLMManager": ("lightspeed_evaluation.core.llm", "LLMManager"),
    "APIClient": ("lightspeed_evaluation.core.api", "APIClient"),
    "OutputHandler": ("lightspeed_evaluation.core.output", "OutputHandler"),
    "GraphGenerator": ("lightspeed_evaluation.core.output", "GraphGenerator"),
    "ScriptExecutionManager": (
        "lightspeed_evaluation.core.script",
        "ScriptExecutionManager",
    ),
    "ScriptExecutionError": (
        "lightspeed_evaluation.core.system.exceptions",
        "ScriptExecutionError",
    ),
    "APIError": ("lightspeed_evaluation.core.system.exceptions", "APIError"),
    "DataValidationError": (
        "lightspeed_evaluation.core.system.exceptions",
        "DataValidationError",
    ),
    "EvaluationError": (
        "lightspeed_evaluation.core.system.exceptions",
        "EvaluationError",
    ),
}

__getattr__ = create_lazy_getattr(_LAZY_IMPORTS, __name__)
__all__ = list(_LAZY_IMPORTS.keys())

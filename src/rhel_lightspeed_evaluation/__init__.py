"""RHEL Lightspeed Evaluation Extensions

This package provides extensions to the lightspeed-evaluation framework,
including:
- Panel of Judges for multi-model evaluation
- Config extensions to support panel of judges

Main components:
- EvaluationPipelineExt: Extended evaluation pipeline with custom processing
- Extended metrics: DeepEval, GEval, and custom metric implementations
- Extended models: SystemConfigExt, EvaluationResultExt
- Extended processors: ConversationProcessorExt, MetricsEvaluatorExt
- Extended output handlers: OutputHandlerExt
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

__version__ = "0.1.0"

__author__ = "Arin DeLoatch"

# Module path constants to avoid duplication
_RHEL_CORE_METRICS = "rhel_lightspeed_evaluation.extensions.core.metrics"
_RHEL_CORE_MODELS = "rhel_lightspeed_evaluation.extensions.core.models"

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
        _RHEL_CORE_MODELS,
        "SystemConfigExt",
    ),
    # Extended models
    "EvaluationResultExt": (
        _RHEL_CORE_MODELS,
        "EvaluationResultExt",
    ),
    # Extended metrics
    "DeepEvalMetricsExt": (
        _RHEL_CORE_METRICS,
        "DeepEvalMetricsExt",
    ),
    "CustomMetricsExt": (
        _RHEL_CORE_METRICS,
        "CustomMetricsExt",
    ),
    "GEvalHandlerExt": (
        _RHEL_CORE_METRICS,
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
}

__getattr__ = create_lazy_getattr(_LAZY_IMPORTS, __name__)

__all__ = [
    # Extended pipeline
    "EvaluationPipelineExt",
    "ConversationProcessorExt",
    "MetricsEvaluatorExt",
    # Extended system config
    "ConfigLoaderExt",
    "SystemConfigExt",
    # Extended models
    "EvaluationResultExt",
    # Extended metrics
    "DeepEvalMetricsExt",
    "CustomMetricsExt",
    "GEvalHandlerExt",
    # Extended LLM
    "DeepEvalLLMManagerExt",
    # Extended output handling
    "OutputHandlerExt",
]

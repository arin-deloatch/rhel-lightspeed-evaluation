"""RHEL Lightspeed Evaluation Extensions

This package provides extensions to the lightspeed-evaluation framework:
- chat/completions API endpoint support via APIClientExt and APIConfigExt
- Extended pipeline and config loader for RHEL-specific API integration
"""

from typing import TYPE_CHECKING

from lightspeed_evaluation.core.system.lazy_import import create_lazy_getattr

if TYPE_CHECKING:
    # ruff: noqa: F401
    from rhel_lightspeed_evaluation.extensions.core.api import APIClientExt
    from rhel_lightspeed_evaluation.extensions.core.models import (
        APIConfigExt,
        SystemConfigExt,
    )
    from rhel_lightspeed_evaluation.extensions.core.system import ConfigLoaderExt
    from rhel_lightspeed_evaluation.extensions.pipeline.evaluation import (
        EvaluationPipelineExt,
    )

__version__ = "0.3.1"

__author__ = "Arin DeLoatch"

_LAZY_IMPORTS = {
    "EvaluationPipelineExt": (
        "rhel_lightspeed_evaluation.extensions.pipeline.evaluation.pipeline",
        "EvaluationPipelineExt",
    ),
    "ConfigLoaderExt": (
        "rhel_lightspeed_evaluation.extensions.core.system",
        "ConfigLoaderExt",
    ),
    "SystemConfigExt": (
        "rhel_lightspeed_evaluation.extensions.core.models",
        "SystemConfigExt",
    ),
    "APIConfigExt": (
        "rhel_lightspeed_evaluation.extensions.core.models",
        "APIConfigExt",
    ),
    "APIClientExt": (
        "rhel_lightspeed_evaluation.extensions.core.api",
        "APIClientExt",
    ),
}

__getattr__ = create_lazy_getattr(_LAZY_IMPORTS, __name__)

__all__ = [
    "EvaluationPipelineExt",
    "ConfigLoaderExt",
    "SystemConfigExt",
    "APIConfigExt",
    "APIClientExt",
]

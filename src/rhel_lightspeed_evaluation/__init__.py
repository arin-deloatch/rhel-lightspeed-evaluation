"""RHEL Lightspeed Evaluation Extensions.

This package provides extensions to the lightspeed-evaluation framework,
including:
- Panel of Judges for multi-model evaluation
- GEval integration for custom evaluation criteria
"""

__version__ = "0.1.0"

from rhel_lightspeed_evaluation.extensions.geval import GEvalMetrics
from rhel_lightspeed_evaluation.extensions.panel_judges import PanelLLMManager

__all__ = ["PanelLLMManager", "GEvalMetrics"]
__author__ = "Arin DeLoatch"

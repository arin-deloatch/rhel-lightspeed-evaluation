"""Framework extensions for lightspeed-evaluation."""

from rhel_lightspeed_evaluation.extensions.geval import GEvalMetrics
from rhel_lightspeed_evaluation.extensions.panel_judges import PanelLLMManager

__all__ = ["PanelLLMManager", "GEvalMetrics"]

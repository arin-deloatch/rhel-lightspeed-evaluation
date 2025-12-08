"""DeepEval metrics evaluation using LLM Manager.

This module provides integration with DeepEval metrics including:
1. Standard DeepEval metrics (conversation completeness, relevancy, knowledge retention)
2. GEval integration for configurable custom evaluation criteria
"""

import logging
from typing import Any, Optional


from rhel_lightspeed_evaluation.extensions.core.llm.deepeval import DeepEvalLLMManagerExt
from lightspeed_evaluation.core.llm.manager import LLMManager
from lightspeed_evaluation.core.metrics.deepeval import DeepEvalMetrics
from lightspeed_evaluation.core.metrics.manager import MetricManager
from lightspeed_evaluation.core.models import EvaluationScope, TurnData

from rhel_lightspeed_evaluation.extensions.core.metrics.geval import GEvalHandlerExt

logger = logging.getLogger(__name__)


class DeepEvalMetricsExt(DeepEvalMetrics):  # pylint: disable=too-few-public-methods
    """Handles DeepEval metrics evaluation using LLM Manager.

    This class provides a unified interface for both standard DeepEval metrics
    and GEval (configurable custom metrics). It shares LLM resources between
    both evaluation types for efficiency.
    """
    def __init__(self,
                 llm_manager: LLMManager,
                 metric_manager: MetricManager,
                 panel_config: Optional[Any]=None):
        """Initialize with LLM Manager
 
        Args:
             llm_manager: Pre-configured LLMManager with validated parameters
             metric_manager: MetricManager for accessing metric metadata
           panel_config: Optional PanelOfJudgesConfig from system configuration
        """
        super().__init__(llm_manager,metric_manager)

        # Pass panel_config to enable panel of judges if configured
        self.llm_manager = DeepEvalLLMManagerExt(
            llm_manager.get_model_name(),
            llm_manager.get_llm_params(),
            panel_config=panel_config,
         )
        
        self.geval_handler = GEvalHandlerExt(
            deepeval_llm_manager=self.llm_manager,
            metric_manager=metric_manager,
        )

        
    def evaluate(
        self,
        metric_name: str,
        conv_data: Any,
        scope: EvaluationScope,
    ) -> list[tuple[str, Optional[float], str]]:
        """Evaluate a DeepEval metric (standard or GEval).

        This method routes evaluation to either:
        - Standard DeepEval metrics (hardcoded implementations)
        - GEval metrics (configuration-driven custom metrics)

        When panel of judges is enabled for GEval metrics, returns multiple results.

        Args:
            metric_name: Name of metric (for GEval, this should NOT include "geval:" prefix)
            conv_data: Conversation data object
            scope: EvaluationScope containing turn info and conversation flag

        Returns:
            List of tuples: [(judge_id, score, reason), ...]
            For standard metrics and non-panel mode, returns single result with judge_id="primary"
        """
        # Route to standard DeepEval metrics
        if metric_name in self.supported_metrics:
            try:
                score, reason = self.supported_metrics[metric_name](
                    conv_data, scope.turn_idx, scope.turn_data, scope.is_conversation
                )
                # Standard metrics always return single result
                return [("primary", score, reason)]
            except (ValueError, AttributeError, KeyError) as e:
                return [
                    (
                        "primary",
                        None,
                        f"DeepEval {metric_name} evaluation failed: {str(e)}",
                    )
                ]

        # Otherwise, assume it's a GEval metric
        normalized_metric_name = (
            metric_name.split(":", 1)[1]
            if metric_name.startswith("geval:")
            else metric_name
        )
        # GEval handler now returns list of results (one per judge, or single for non-panel)
        return self.geval_handler.evaluate(
            metric_name=normalized_metric_name,
            conv_data=conv_data,
            _turn_idx=scope.turn_idx,
            turn_data=scope.turn_data,
            is_conversation=scope.is_conversation,
        )
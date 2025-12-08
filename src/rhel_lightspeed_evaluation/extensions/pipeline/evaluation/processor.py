"""Metrics evaluation module - handles individual metric evaluation."""

import logging
from lightspeed_evaluation.pipeline.evaluation import ConversationProcessor
from lightspeed_evaluation.core.models import (
    EvaluationData,
    EvaluationRequest,
    EvaluationResult,
    TurnData,
)

logger = logging.getLogger(__name__)


class ConversationProcessorExt(ConversationProcessor):
    def _evaluate_turn(
        self,
        conv_data: EvaluationData,
        turn_idx: int,
        turn_data: TurnData,
        turn_metrics: list[str],
    ) -> list[EvaluationResult]:
        """Evaluate single turn with specified turn metrics."""
        results = []

        for metric_identifier in turn_metrics:
            request = EvaluationRequest.for_turn(
                conv_data, metric_identifier, turn_idx, turn_data
            )
            result = self.components.metrics_evaluator.evaluate_metric(request)
            if result:
                # evaluate_metric returns list[EvaluationResult], so extend not append
                results.extend(result)
        return results

    def _evaluate_conversation(
        self, conv_data: EvaluationData, conversation_metrics: list[str]
    ) -> list[EvaluationResult]:
        """Evaluate conversation-level metrics."""
        results = []

        for metric_identifier in conversation_metrics:
            request = EvaluationRequest.for_conversation(conv_data, metric_identifier)
            result = self.components.metrics_evaluator.evaluate_metric(request)
            if result:
                # evaluate_metric returns list[EvaluationResult], so extend not append
                results.extend(result)
        return results
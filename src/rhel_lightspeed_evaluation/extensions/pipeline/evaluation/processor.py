"""Preserve per-metric failure recovery on the deployment-pinned upstream."""

import logging

from lightspeed_evaluation.core.models import (
    EvaluationData,
    EvaluationRequest,
    EvaluationResult,
    TurnData,
)
from lightspeed_evaluation.pipeline.evaluation.processor import ConversationProcessor

logger = logging.getLogger(__name__)


class ConversationProcessorExt(ConversationProcessor):
    """Backport the behavior of upstream PR #304 without changing its package."""

    # shortcut: remove this override once lscore-deploy pins a revision containing PR #304.
    def _evaluate_safely(self, request: EvaluationRequest) -> list[EvaluationResult]:
        try:
            if request.is_conversation:
                return super()._evaluate_conversation(
                    request.conv_data, [request.metric_identifier]
                )
            assert request.turn_data is not None and request.turn_idx is not None
            return super()._evaluate_turn(
                request.conv_data, request.turn_idx, request.turn_data, [request.metric_identifier]
            )
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            logger.error("%s evaluation failed: %s", request.metric_identifier, reason)
            return [
                self.components.error_handler.create_error_result(
                    request.conv_data.conversation_group_id,
                    request.metric_identifier,
                    reason,
                    tag=request.conv_data.tag,
                    turn_id=request.turn_id,
                    query=(request.turn_data.query or "") if request.turn_data else "",
                )
            ]

    def _evaluate_turn(
        self,
        conv_data: EvaluationData,
        turn_idx: int,
        turn_data: TurnData,
        turn_metrics: list[str],
    ) -> list[EvaluationResult]:
        results = []
        for metric in turn_metrics:
            results.extend(
                self._evaluate_safely(
                    EvaluationRequest.for_turn(conv_data, metric, turn_idx, turn_data)
                )
            )
        return results

    def _evaluate_conversation(
        self, conv_data: EvaluationData, conversation_metrics: list[str]
    ) -> list[EvaluationResult]:
        results = []
        for metric in conversation_metrics:
            results.extend(
                self._evaluate_safely(EvaluationRequest.for_conversation(conv_data, metric))
            )
        return results

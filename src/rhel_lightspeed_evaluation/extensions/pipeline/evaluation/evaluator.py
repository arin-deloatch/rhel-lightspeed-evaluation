from lightspeed_evaluation.pipeline.evaluation import MetricsEvaluator
import logging
import time
from typing import Optional

from rhel_lightspeed_evaluation.extensions.core.metrics.deepeval import DeepEvalMetricsExt
from rhel_lightspeed_evaluation.extensions.core.models.data import EvaluationResultExt
from rhel_lightspeed_evaluation.extensions.core.system.loader import ConfigLoaderExt

from lightspeed_evaluation.core.embedding.manager import EmbeddingManager
from lightspeed_evaluation.core.llm.manager import LLMManager
from lightspeed_evaluation.core.metrics.custom import CustomMetrics
from lightspeed_evaluation.core.metrics.manager import MetricLevel, MetricManager
from lightspeed_evaluation.core.metrics.ragas import RagasMetrics
from lightspeed_evaluation.core.metrics.script import ScriptEvalMetrics
from lightspeed_evaluation.core.models import (
    EvaluationRequest,
    EvaluationScope,
)
from lightspeed_evaluation.core.script import ScriptExecutionManager

logger = logging.getLogger(__name__)

class MetricsEvaluatorExt(MetricsEvaluator):
    """Metrics evaluation module - handles individual metric evaluation."""
    def __init__(
        self,
        config_loader: ConfigLoaderExt,
        metric_manager: MetricManager,
        script_manager: ScriptExecutionManager,
    ) -> None:
        """Initialize Metric Evaluator."""
        self.config_loader = config_loader
        if config_loader.system_config is None:
            raise RuntimeError("Uninitialized system_config")

        llm_manager = LLMManager.from_system_config(config_loader.system_config)
        embedding_manager = EmbeddingManager.from_system_config(
            config_loader.system_config
        )

        # Initialize metric handlers
        self.ragas_metrics = RagasMetrics(llm_manager, embedding_manager)
        self.deepeval_metrics = DeepEvalMetricsExt(
            llm_manager,
            metric_manager=metric_manager,
            panel_config=config_loader.system_config.panel_of_judges,
        )
        self.custom_metrics = CustomMetrics(llm_manager)
        self.script_eval_metrics = ScriptEvalMetrics(script_manager)

        # Metric routing map
        self.handlers = {
            "ragas": self.ragas_metrics,
            "deepeval": self.deepeval_metrics,
            # Note: geval metrics are routed through deepeval_metrics handler
            "geval": self.deepeval_metrics,
            "custom": self.custom_metrics,
            "script": self.script_eval_metrics,
        }

        self.metric_manager = metric_manager

    def evaluate_metric(  # pylint: disable=R0914
        self, request: EvaluationRequest
    ) -> list[EvaluationResultExt] | None:
        """Evaluate a single metric and return result(s).

        For panel of judges evaluations, returns multiple results (one per judge).
        For standard evaluations, returns a single-item list.

        Returns:
            List of EvaluationResultExt objects, or None if evaluation should be skipped
        """
        start_time = time.time()

        try:
            # Create logging summary
            if request.is_conversation:
                summary = (
                    f"Conversation {request.conv_data.conversation_group_id} - "
                    f"{request.metric_identifier}"
                )
            else:
                summary = f"Turn {request.turn_id} - {request.metric_identifier}"
            logger.debug("Evaluating: %s", summary)

            # Parse framework and metric
            framework, metric_name = request.metric_identifier.split(":", 1)

            # Skip script metrics if API is disabled
            if (
                framework == "script"
                and self.config_loader.system_config is not None
                and not self.config_loader.system_config.api.enabled
            ):
                # Don't generate result for script metrics when API disabled
                return None

            # Route to appropriate handler
            if framework not in self.handlers:
                execution_time = time.time() - start_time
                return [
                    self._create_error_result(
                        request,
                        f"Unsupported framework: {framework}",
                        execution_time,
                        None,
                    )
                ]

            # Create evaluation scope
            evaluation_scope = EvaluationScope(
                turn_idx=request.turn_idx,
                turn_data=request.turn_data,
                is_conversation=request.is_conversation,
            )

            # Evaluate metric
            handler_result = self.handlers[framework].evaluate(  # type: ignore
                metric_name, request.conv_data, evaluation_scope
            )

            execution_time = time.time() - start_time

            # Get threshold once (same for all judges)
            level = (
                MetricLevel.CONVERSATION
                if request.is_conversation
                else MetricLevel.TURN
            )
            threshold = self.metric_manager.get_effective_threshold(
                request.metric_identifier, level, request.conv_data, request.turn_data
            )

            # Handle different return types:
            # - deepeval, geval, custom: list[tuple[str, score, reason]]
            # - ragas, script: tuple[score, reason]
            if isinstance(handler_result, list):
                # Panel-aware handlers (deepeval, geval, custom)
                results = []
                for judge_id, score, reason in handler_result:
                    if score is None:
                        results.append(
                            self._create_error_result(
                                request, reason, execution_time, judge_id
                            )
                        )
                    else:
                        status = self._determine_status(score, threshold)
                        results.append(
                            EvaluationResultExt(
                                conversation_group_id=request.conv_data.conversation_group_id,
                                turn_id=request.turn_id,
                                metric_identifier=request.metric_identifier,
                                judge_id=judge_id if judge_id != "primary" else None,
                                result=status,
                                score=score,
                                threshold=threshold,
                                reason=reason,
                                query=(
                                    request.turn_data.query if request.turn_data else ""
                                ),
                                response=(
                                    request.turn_data.response or ""
                                    if request.turn_data
                                    else ""
                                ),
                                execution_time=execution_time,
                            )
                        )
                return results

            # Legacy handlers (ragas, script) - return single result
            score, reason = handler_result
            if score is None:
                return [
                    self._create_error_result(request, reason, execution_time, None)
                ]

            status = self._determine_status(score, threshold)
            return [
                EvaluationResultExt(
                    conversation_group_id=request.conv_data.conversation_group_id,
                    turn_id=request.turn_id,
                    metric_identifier=request.metric_identifier,
                    judge_id=None,
                    result=status,
                    score=score,
                    threshold=threshold,
                    reason=reason,
                    query=request.turn_data.query if request.turn_data else "",
                    response=(
                        request.turn_data.response or "" if request.turn_data else ""
                    ),
                    execution_time=execution_time,
                )
            ]

        except Exception as e:  # pylint: disable=broad-exception-caught
            # Any evaluation error should result in ERROR status
            execution_time = time.time() - start_time
            return [
                self._create_error_result(
                    request, f"Evaluation error: {e}", execution_time, None
                )
            ]

    def _create_error_result(
        self,
        request: EvaluationRequest,
        reason: str,
        execution_time: float,
        judge_id: str | None,
    ) -> EvaluationResultExt:
        """Create an ERROR result for failed evaluation."""
        return EvaluationResultExt(
            conversation_group_id=request.conv_data.conversation_group_id,
            turn_id=request.turn_id,
            metric_identifier=request.metric_identifier,
            judge_id=judge_id if judge_id != "primary" else None,
            result="ERROR",
            score=None,
            threshold=None,
            reason=reason,
            query=request.turn_data.query if request.turn_data else "",
            response=request.turn_data.response or "" if request.turn_data else "",
            execution_time=execution_time,
        )

    def _determine_status(self, score: float, threshold: Optional[float]) -> str:
        """Determine evaluation status based on score and threshold."""
        if threshold is None:
            threshold = 0.5  # This will also handle binary metrics
        return "PASS" if score >= float(threshold) else "FAIL"

    def get_supported_frameworks(self) -> list[str]:
        """Get list of supported evaluation frameworks."""
        return list(self.handlers.keys())

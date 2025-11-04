"""GEval integration for runtime-configurable custom evaluation criteria.

This module provides integration with DeepEval's GEval metric that allows
defining custom evaluation criteria, parameters, and steps via:
1. Centralized metric registry (config/geval_metrics.yaml) - recommended
2. Runtime YAML configuration - for one-off custom metrics

Usage with metric registry (recommended):
    turns:
      - turn_id: "turn_1"
        query: "How do I check firewall status?"
        response: "Use systemctl status firewalld"

        turn_metrics:
          - "geval:technical_accuracy"
          - "geval:command_safety"
        # No metadata needed! Metrics loaded from registry.

Usage with runtime configuration (for custom metrics):
    turns:
      - turn_id: "turn_1"
        query: "How do I check firewall status?"
        response: "Use systemctl status firewalld"

        turn_metrics:
          - "geval:custom_metric"

        turn_metrics_metadata:
          geval:custom_metric:
            criteria: "Your custom criteria..."
            evaluation_params: [...]
            evaluation_steps: [...]
            threshold: 0.8
"""

import logging
from pathlib import Path
from typing import Any, Optional

import yaml
from deepeval.metrics import GEval
from deepeval.test_case import ConversationalTestCase, LLMTestCase, LLMTestCaseParams

from lightspeed_evaluation.core.llm.deepeval import DeepEvalLLMManager
from lightspeed_evaluation.core.llm.manager import LLMManager

logger = logging.getLogger(__name__)


class GEvalHandler:
    """Handler for runtime-configurable GEval metrics.

    This class integrates with the lightspeed-evaluation framework
    to provide GEval evaluation with criteria defined either in:
    1. A centralized metric registry (config/geval_metrics.yaml)
    2. Runtime YAML configuration (turn_metrics_metadata)

    Priority: Runtime metadata overrides registry definitions.
    """

    # Class-level registry cache (shared across instances)
    _registry: Optional[dict[str, Any]] = None
    _registry_path: Optional[Path] = None

    def __init__(
        self,
        llm_manager: LLMManager,
        registry_path: Optional[str] = None,
    ) -> None:
        """Initialize GEval handler.

        Args:
            llm_manager: LLM manager from lightspeed-evaluation framework
            registry_path: Optional path to metric registry YAML.
                          If not provided, looks for config/geval_metrics.yaml
                          relative to project root.
        """
        # Create DeepEval LLM Manager wrapper for GEval metrics
        # This provides the get_llm() method that returns a LiteLLMModel
        self.deepeval_llm_manager = DeepEvalLLMManager(
            llm_manager.get_model_name(), llm_manager.get_llm_params()
        )
        self._load_registry(registry_path)

    def _load_registry(self, registry_path: Optional[str] = None) -> None:
        """Load metric registry from YAML file.

        Args:
            registry_path: Optional path to registry file
        """
        # Only load once per class
        if GEvalHandler._registry is not None:
            return

        # Determine registry path
        if registry_path:
            path = Path(registry_path)
        else:
            # Look for config/geval_metrics.yaml relative to project root
            # Try multiple locations
            possible_paths = [
                Path.cwd() / "config" / "registry" / "geval_metrics.yaml",
                Path(__file__).parent.parent.parent.parent / "config" / "registry" / "geval_metrics.yaml",
            ]
            path = None
            for p in possible_paths:
                if p.exists():
                    path = p
                    break

        if path is None or not path.exists():
            logger.warning(
                f"GEval metric registry not found at expected locations. "
                f"Tried: {[str(p) for p in possible_paths]}. "
                f"Will fall back to runtime metadata only."
            )
            GEvalHandler._registry = {}
            return

        try:
            with open(path) as f:
                GEvalHandler._registry = yaml.safe_load(f) or {}
                GEvalHandler._registry_path = path
                num_metrics = len(GEvalHandler._registry) if GEvalHandler._registry else 0
                logger.info(f"Loaded {num_metrics} GEval metrics from {path}")
        except Exception as e:
            logger.error(f"Failed to load GEval registry from {path}: {e}")
            GEvalHandler._registry = {}

    def evaluate(
        self,
        metric_name: str,
        conv_data: Any,
        turn_idx: Optional[int],  # noqa: ARG002
        turn_data: Optional[Any],
        is_conversation: bool,
    ) -> tuple[Optional[float], str]:
        """Evaluate using GEval with runtime configuration.

        This method extracts GEval configuration from metadata and
        performs evaluation.

        Args:
            metric_name: Name of the metric (e.g., "technical_accuracy")
            conv_data: Conversation data object
            turn_idx: Turn index (unused, kept for interface compatibility)
            turn_data: Turn data object (for turn-level metrics)
            is_conversation: Whether this is conversation-level evaluation

        Returns:
            Tuple of (score, reason)
        """
        # Extract GEval configuration from metadata
        geval_config = self._get_geval_config(
            metric_name, conv_data, turn_data, is_conversation
        )

        if not geval_config:
            return None, f"GEval configuration not found for metric '{metric_name}'"

        # Extract configuration parameters
        criteria = geval_config.get("criteria")
        evaluation_params = geval_config.get("evaluation_params", [])
        evaluation_steps = geval_config.get("evaluation_steps")
        threshold = geval_config.get("threshold", 0.5)

        if not criteria:
            return None, "GEval requires 'criteria' in configuration"

        # Perform evaluation based on level
        if is_conversation:
            return self._evaluate_conversation(
                conv_data, criteria, evaluation_params, evaluation_steps, threshold
            )
        else:
            return self._evaluate_turn(
                turn_data, criteria, evaluation_params, evaluation_steps, threshold
            )

    def _convert_evaluation_params(
        self, params: list[str]
    ) -> Optional[list[LLMTestCaseParams]]:
        """Convert string params to LLMTestCaseParams enum values.

        Args:
            params: List of parameter strings

        Returns:
            List of LLMTestCaseParams enum values, or None if params are custom strings
        """
        if not params:
            return None

        # Try to convert strings to enum values
        converted = []
        for param in params:
            try:
                # Try to match as enum value (e.g., "INPUT", "ACTUAL_OUTPUT")
                enum_value = LLMTestCaseParams[param.upper().replace(" ", "_")]
                converted.append(enum_value)
            except (KeyError, AttributeError):
                # Not a valid enum - these are custom params, skip them
                logger.debug(
                    f"Skipping custom evaluation_param '{param}' - "
                    f"not a valid LLMTestCaseParams enum. "
                    f"GEval will auto-detect required fields."
                )
                return None

        return converted if converted else None

    def _evaluate_turn(
        self,
        turn_data: Any,
        criteria: str,
        evaluation_params: list[str],
        evaluation_steps: Optional[list[str]],
        threshold: float,
    ) -> tuple[Optional[float], str]:
        """Evaluate a single turn using GEval.

        Args:
            turn_data: Turn data object
            criteria: Evaluation criteria description
            evaluation_params: List of evaluation parameters (strings)
            evaluation_steps: Optional list of evaluation steps
            threshold: Score threshold

        Returns:
            Tuple of (score, reason)
        """
        if not turn_data:
            return None, "Turn data required for turn-level GEval"

        # Convert evaluation_params to enum values if valid, otherwise use defaults
        converted_params = self._convert_evaluation_params(evaluation_params)
        if not converted_params:
            # If no valid params, use sensible defaults for turn evaluation
            converted_params = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]

        # Create GEval metric with runtime configuration
        metric_kwargs: dict[str, Any] = {
            "name": "GEval Turn Metric",
            "criteria": criteria,
            "evaluation_params": converted_params,
            "model": self.deepeval_llm_manager.get_llm(),
            "threshold": threshold,
            "top_logprobs": 5,  # Set to safe value for all LLM providers (Vertex AI limit is 19)
        }

        # Add evaluation steps if provided
        if evaluation_steps:
            metric_kwargs["evaluation_steps"] = evaluation_steps

        metric = GEval(**metric_kwargs)

        # Create test case
        test_case = LLMTestCase(
            input=turn_data.query,
            actual_output=turn_data.response or "",
            expected_output=turn_data.expected_response,
            context=turn_data.contexts,
        )

        # Evaluate
        try:
            metric.measure(test_case)
            score = metric.score if metric.score is not None else 0.0
            reason = str(metric.reason) if hasattr(metric, "reason") and metric.reason else "No reason provided"
            return score, reason
        except Exception as e:
            return None, f"GEval evaluation error: {str(e)}"

    def _evaluate_conversation(
        self,
        conv_data: Any,
        criteria: str,
        evaluation_params: list[str],
        evaluation_steps: Optional[list[str]],
        threshold: float,
    ) -> tuple[Optional[float], str]:
        """Evaluate a conversation using GEval.

        Args:
            conv_data: Conversation data object
            criteria: Evaluation criteria description
            evaluation_params: List of evaluation parameters (strings)
            evaluation_steps: Optional list of evaluation steps
            threshold: Score threshold

        Returns:
            Tuple of (score, reason)
        """
        # Convert evaluation_params to enum values if valid, otherwise use defaults
        converted_params = self._convert_evaluation_params(evaluation_params)
        if not converted_params:
            # If no valid params, use sensible defaults for conversation evaluation
            converted_params = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]

        # Create GEval metric with runtime configuration
        metric_kwargs: dict[str, Any] = {
            "name": "GEval Conversation Metric",
            "criteria": criteria,
            "evaluation_params": converted_params,
            "model": self.deepeval_llm_manager.get_llm(),
            "threshold": threshold,
            "top_logprobs": 5,  # Set to safe value for all LLM providers (Vertex AI limit is 19)
        }

        # Add evaluation steps if provided
        if evaluation_steps:
            metric_kwargs["evaluation_steps"] = evaluation_steps

        metric = GEval(**metric_kwargs)

        # Convert turns to DeepEval format
        llm_test_cases = []
        for turn in conv_data.turns:
            llm_test_cases.append(
                LLMTestCase(
                    input=turn.query,
                    actual_output=turn.response or "",
                )
            )

        test_case = ConversationalTestCase(turns=llm_test_cases)

        # Evaluate
        try:
            metric.measure(test_case)  # type: ignore[arg-type]
            score = metric.score if metric.score is not None else 0.0
            reason = str(metric.reason) if hasattr(metric, "reason") and metric.reason else "No reason provided"
            return score, reason
        except Exception as e:
            return None, f"GEval evaluation error: {str(e)}"

    def _get_geval_config(
        self,
        metric_name: str,
        conv_data: Any,
        turn_data: Optional[Any],
        is_conversation: bool,
    ) -> Optional[dict[str, Any]]:
        """Extract GEval configuration from metadata or registry.

        Priority order (highest to lowest):
        1. Turn-level metadata (for turn metrics) - runtime override
        2. Conversation-level metadata - runtime override
        3. Metric registry (config/geval_metrics.yaml) - shared definitions

        Args:
            metric_name: Name of the metric
            conv_data: Conversation data object
            turn_data: Turn data object
            is_conversation: Whether this is conversation-level evaluation

        Returns:
            Configuration dictionary or None
        """
        metric_key = f"geval:{metric_name}"

        # Priority 1: Check turn-level metadata (runtime override)
        if not is_conversation and turn_data and hasattr(turn_data, "turn_metrics_metadata"):
            if turn_data.turn_metrics_metadata and metric_key in turn_data.turn_metrics_metadata:
                logger.debug(f"Using runtime metadata for metric '{metric_name}'")
                return turn_data.turn_metrics_metadata[metric_key]

        # Priority 2: Check conversation-level metadata (runtime override)
        if hasattr(conv_data, "conversation_metrics_metadata"):
            if conv_data.conversation_metrics_metadata and metric_key in conv_data.conversation_metrics_metadata:
                logger.debug(f"Using runtime metadata for metric '{metric_name}'")
                return conv_data.conversation_metrics_metadata[metric_key]

        # Priority 3: Check metric registry
        if GEvalHandler._registry and metric_name in GEvalHandler._registry:
            logger.debug(f"Using registry definition for metric '{metric_name}'")
            return GEvalHandler._registry[metric_name]

        # Not found anywhere
        logger.warning(
            f"Metric '{metric_name}' not found in runtime metadata or registry. "
            f"Available registry metrics: {list(GEvalHandler._registry.keys()) if GEvalHandler._registry else []}"
        )
        return None


class GEvalMetrics:
    """GEval metrics handler for lightspeed-evaluation framework.

    This class implements the metric handler interface expected by
    MetricsEvaluator, allowing GEval metrics to be used alongside
    ragas, deepeval, custom, and script metrics.
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        registry_path: Optional[str] = None,
    ) -> None:
        """Initialize GEval metrics handler.

        Args:
            llm_manager: LLM manager from lightspeed-evaluation framework
            registry_path: Optional path to metric registry YAML
        """
        self.handler = GEvalHandler(llm_manager, registry_path)

    def evaluate(
        self,
        metric_name: str,
        conv_data: Any,
        evaluation_scope: Any,
    ) -> tuple[Optional[float], str]:
        """Evaluate a GEval metric.

        This method implements the standard metric handler interface
        expected by MetricsEvaluator.

        Args:
            metric_name: Name of the metric (without "geval:" prefix)
            conv_data: Conversation data object
            evaluation_scope: EvaluationScope with turn_idx, turn_data, is_conversation

        Returns:
            Tuple of (score, reason)
        """
        return self.handler.evaluate(
            metric_name=metric_name,
            conv_data=conv_data,
            turn_idx=evaluation_scope.turn_idx,
            turn_data=evaluation_scope.turn_data,
            is_conversation=evaluation_scope.is_conversation,
        )

"""
GEval integration for configurable custom evaluation criteria.

This module provides integration with DeepEval's GEval metric that allows
defining custom evaluation criteria, parameters, and steps via:
1. Centralized metric registry (config/registry/geval_metrics.yaml)
2. Runtime YAML configuration
"""

import logging
from pathlib import Path
from typing import Any

import yaml
from deepeval.metrics import GEval
from deepeval.test_case import ConversationalTestCase, LLMTestCase, LLMTestCaseParams
from lightspeed_evaluation.core.llm.deepeval import DeepEvalLLMManager
from lightspeed_evaluation.core.llm.manager import LLMManager

logger = logging.getLogger(__name__)


class GEvalHandler:
    """Handler for configurable GEval metrics.

    This class integrates with the lightspeed-evaluation framework
    to provide GEval evaluation with criteria defined either in:
    1. A centralized metric registry (config/geval_metrics.yaml)
    2. Runtime YAML configuration (turn_metrics_metadata)

    Priority: Runtime metadata overrides registry definitions.
    """

    # Class-level registry cache (shared across instances)
    _registry: dict[str, Any] | None = None
    _registry_path: Path | None = None

    def __init__(
        self,
        llm_manager: LLMManager,
        registry_path: str | None = None,
    ) -> None:
        """Initialize GEval handler.

        Args:
            llm_manager: LLM manager from lightspeed-evaluation framework
            registry_path: Optional path to metric registry YAML.
                          If not provided, looks for config/registry/geval_metrics.yaml
                          relative to project root.
        """
        # Create DeepEval LLM Manager wrapper for GEval metrics
        self.deepeval_llm_manager = DeepEvalLLMManager(
            llm_manager.get_model_name(), llm_manager.get_llm_params()
        )
        self._load_registry(registry_path)

    def _load_registry(self, registry_path: str | None = None) -> None:
        """
        Load the GEval metric registry from a YAML configuration file.

        This method initializes the class-level `_registry`.
        It supports both user-specified and auto-discovered paths, searching common
        locations relative to the current working directory and the package root.

        If no valid registry file is found, it logs a warning and initializes an
        empty registry (meaning GEval will rely solely on runtime metadata).

        Args:
            registry_path (str | None): Optional explicit path to a registry YAML file.

        Behavior:
            - If the registry has already been loaded, the function returns immediately.
            - If `registry_path` is provided, it is used directly.
            - Otherwise, common fallback paths are checked for existence.
            - If a registry is found, it is parsed with `yaml.safe_load`.
            - Any exceptions during file access or parsing are logged, and an empty
            registry is used as a fallback.
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
                Path(__file__).parent.parent.parent.parent
                / "config"
                / "registry"
                / "geval_metrics.yaml",
            ]
            path = None
            for p in possible_paths:
                if p.exists():
                    path = p
                    break
        # Handle missing or invalid registry
        if path is None or not path.exists():
            logger.warning(
                f"GEval metric registry not found at expected locations. "
                f"Tried: {[str(p) for p in possible_paths]}. "
                f"Will fall back to runtime metadata only."
            )
            GEvalHandler._registry = {}
            return

        # Load registry file
        try:
            with open(path) as f:
                GEvalHandler._registry = yaml.safe_load(f) or {}  # Check in system config
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
        turn_idx: int | None,  # noqa: ARG002
        turn_data: Any | None,
        is_conversation: bool,
    ) -> tuple[float | None, str]:
        """
        Evaluate using GEval with runtime configuration.

        This method is the central entry point for running GEval evaluations.
        It retrieves the appropriate metric configuration (from registry or runtime
        metadata), extracts evaluation parameters, and delegates the actual scoring
        to either conversation-level or turn-level evaluators.

         Args:
            metric_name (str):
                The name of the metric to evaluate (e.g., "technical_accuracy").
            conv_data (Any):
                The conversation data object containing context, messages, and
                associated metadata.
            turn_idx (int | None):
                The index of the current turn in the conversation.
                (Currently unused but kept for interface compatibility.)
            turn_data (Any | None):
                The turn-level data object, required when evaluating turn-level metrics.
            is_conversation (bool):
                Indicates whether the evaluation should run on the entire
                conversation (`True`) or on an individual turn (`False`).

        Returns:
        tuple[float | None, str]:
            A tuple containing:
              - **score** (float | None): The computed metric score, or None if evaluation failed.
              - **reason** (str): A descriptive reason or error message.

        Behavior:
        1. Fetch GEval configuration from metadata using `_get_geval_config()`.
        2. Validate that required fields (e.g., "criteria") are present.
        3. Extract key parameters such as criteria, evaluation steps, and threshold.
        4. Delegate to `_evaluate_conversation()` or `_evaluate_turn()` depending
           on the `is_conversation` flag.
        """
        # Extract GEval configuration from metadata
        # May come from runtime metadata or a preloaded registry
        geval_config = self._get_geval_config(metric_name, conv_data, turn_data, is_conversation)

        # If no configuration is available, return early with an informative message.
        if not geval_config:
            return None, f"GEval configuration not found for metric '{metric_name}'"

        # Extract configuration parameters
        criteria = geval_config.get("criteria")
        evaluation_params = geval_config.get("evaluation_params", [])
        evaluation_steps = geval_config.get("evaluation_steps")
        threshold = geval_config.get("threshold", 0.5)

        # The criteria field defines what the model is being judged on.
        # Without it, we cannot perform evaluation. Evaluation steps can be generated
        if not criteria:
            return None, "GEval requires 'criteria' in configuration"

        # Perform evaluation based on level (turn or conversation)
        if is_conversation:
            return self._evaluate_conversation(
                conv_data, criteria, evaluation_params, evaluation_steps, threshold
            )
        else:
            return self._evaluate_turn(
                turn_data, criteria, evaluation_params, evaluation_steps, threshold
            )

    def _convert_evaluation_params(self, params: list[str]) -> list[LLMTestCaseParams] | None:
        """
        Convert a list of string parameter names into `LLMTestCaseParams` enum values.

        This helper ensures that the evaluation parameters passed into GEval are properly
        typed as `LLMTestCaseParams` (used by DeepEval’s test-case schema). If any parameter is not a
        valid enum member, the function treats the entire parameter list as "custom" and returns `None`.
        This allows GEval to automatically infer the required fields at runtime rather than forcing
        strict schema compliance.

        Args:
            params (list[str]):
                A list of string identifiers (e.g., ["input", "actual_output"]).
                These typically come from a YAML or runtime configuration and
                may not always match enum names exactly.
        Returns:
            List of LLMTestCaseParams enum values, or None if params are custom strings
        """
        # Return early if no parameters were supplied
        if not params:
            return None

        # Try to convert strings to enum values
        converted: list[LLMTestCaseParams] = []

        # Attempt to convert each string into a valid enum value
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

        # Return the successfully converted list, or None if it ended up empty
        return converted if converted else None

    def _evaluate_turn(
        self,
        turn_data: Any,
        criteria: str,
        evaluation_params: list[str],
        evaluation_steps: list[str] | None,
        threshold: float,
    ) -> tuple[float | None, str]:
        """
            Evaluate a single turn using GEval.

            Args:
            turn_data (Any):
                The turn-level data object containing fields like query, response,
                expected_response, and context.
            criteria (str):
                Natural-language description of what the evaluation should judge.
                Example: "Assess factual correctness and command validity."
            evaluation_params (list[str]):
                A list of string parameters defining which fields to include
                (e.g., ["input", "actual_output"]).
            evaluation_steps (list[str] | None):
                Optional step-by-step evaluation guidance for the model.
            threshold (float):
                Minimum score threshold that determines pass/fail behavior.

        Returns:
            tuple[float | None, str]:
                A tuple of (score, reason). If evaluation fails, score will be None
                and the reason will contain an error message.
        """
        # Validate that we actually have turn data
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

        # Instantiate the GEval metric object
        metric = GEval(**metric_kwargs)

        # Create test case for a single turn
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
            reason = (
                str(metric.reason)
                if hasattr(metric, "reason") and metric.reason
                else "No reason provided"
            )
            return score, reason
        except Exception as e:
            return None, f"GEval evaluation error: {str(e)}"

    def _evaluate_conversation(
        self,
        conv_data: Any,
        criteria: str,
        evaluation_params: list[str],
        evaluation_steps: list[str] | None,
        threshold: float,
    ) -> tuple[float | None, str]:
        """
        Evaluate a conversation using GEval.

        This method aggregates all conversation turns into a DeepEval
        ConversationalTestCase and evaluates the conversation against
        the provided criteria.

        Args:
            conv_data (Any):
                Conversation data object containing all turns.
            criteria (str):
                Description of the overall evaluation goal.
            evaluation_params (list[str]):
                List of field names to include (same semantics as turn-level).
            evaluation_steps (list[str] | None):
                Optional instructions guiding how the evaluation should proceed.
            threshold (float):
                Minimum acceptable score before the metric is considered failed.

        Returns:
            tuple[float | None, str]:
                Tuple containing (score, reason). Returns None on error.
        """
        # Convert evaluation_params to enum values if valid, otherwise use defaults
        converted_params = self._convert_evaluation_params(evaluation_params)
        if not converted_params:
            # If no valid params, use sensible defaults for conversation evaluation
            converted_params = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]

        # Configure the GEval metric for conversation-level evaluation
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

        # Instantiate the GEval metric object
        metric = GEval(**metric_kwargs)

        # Convert conversation turns into DeepEval-compatible test cases
        llm_test_cases = []
        for turn in conv_data.turns:
            llm_test_cases.append(
                LLMTestCase(
                    input=turn.query,
                    actual_output=turn.response or "",
                )
            )
        # Wrap the list in a conversational test case
        test_case = ConversationalTestCase(turns=llm_test_cases)

        # Evaluate
        try:
            metric.measure(test_case)  # type: ignore[arg-type]
            score = metric.score if metric.score is not None else 0.0
            reason = (
                str(metric.reason)
                if hasattr(metric, "reason") and metric.reason
                else "No reason provided"
            )
            return score, reason
        except Exception as e:
            return None, f"GEval evaluation error: {str(e)}"

    def _get_geval_config(
        self,
        metric_name: str,
        conv_data: Any,
        turn_data: Any | None,
        is_conversation: bool,
    ) -> dict[str, Any] | None:
        """E
        xtract GEval configuration from metadata or registry.

         The method checks multiple sources in priority order:
            1. Turn-level metadata (runtime override)
            2. Conversation-level metadata (runtime override)
            3. Metric registry (shared, persistent YAML definitions)

         Args:
            metric_name (str):
                Name of the metric to retrieve (e.g., "completeness").
            conv_data (Any):
                The full conversation data object, which may contain
                conversation-level metadata.
            turn_data (Any | None):
                Optional turn-level data object, for per-turn metrics.
            is_conversation (bool):
                True if evaluating a conversation-level metric, False for turn-level.

        Returns:
            dict[str, Any] | None:
                The GEval configuration dictionary if found, otherwise None.
        """
        metric_key = f"geval:{metric_name}"

        # Turn level metadata override
        # Used when individual turns define custom GEval settings
        if (
            not is_conversation
            and turn_data
            and hasattr(turn_data, "turn_metrics_metadata")
            and turn_data.turn_metrics_metadata
            and metric_key in turn_data.turn_metrics_metadata
        ):
            logger.debug(f"Using runtime metadata for metric '{metric_name}'")
            return turn_data.turn_metrics_metadata[metric_key]

        # Conversation-level metadata override
        # Used when the conversation defines shared GEval settings
        if (
            hasattr(conv_data, "conversation_metrics_metadata")
            and conv_data.conversation_metrics_metadata
            and metric_key in conv_data.conversation_metrics_metadata
        ):
            logger.debug(f"Using runtime metadata for metric '{metric_name}'")
            return conv_data.conversation_metrics_metadata[metric_key]

        # Registry definition
        # Fallback to shared YAML registry if no runtime metadata is found
        if GEvalHandler._registry and metric_name in GEvalHandler._registry:
            logger.debug(f"Using registry definition for metric '{metric_name}'")
            return GEvalHandler._registry[metric_name]

        # Config not found anywhere
        logger.warning(
            f"Metric '{metric_name}' not found in runtime metadata or registry. "
            f"Available registry metrics: {list(GEvalHandler._registry.keys()) if GEvalHandler._registry else []}"
        )
        return None


class GEvalMetrics:
    """
    GEval metrics handler for lightspeed-evaluation framework.

    This class implements the metric handler interface expected by
    MetricsEvaluator, allowing GEval metrics to be used alongside
    ragas, deepeval, custom, and script metrics.
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        registry_path: str | None = None,
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
    ) -> tuple[float | None, str]:
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

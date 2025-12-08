"""DeepEval LLM Manager - DeepEval-specific LLM wrapper."""

from typing import Any, Optional

from deepeval.models import LiteLLMModel

from lightspeed_evaluation.core.llm import DeepEvalLLMManager, LLMManager

from lightspeed_evaluation.core.models import LLMConfig

from rhel_lightspeed_evaluation.extensions.core.models.system import PanelOfJudgesConfig, JudgeConfig

class DeepEvalLLMManagerExt(DeepEvalLLMManager):
    """DeepEval LLM Manager - Takes LLM parameters directly.

    This manager focuses solely on DeepEval-specific LLM integration.
    Supports both single-LLM mode and panel of judges mode.

    Args:
        model_name: Primary LLM model name
        llm_params: LLM parameters (temperature, max_tokens, etc.)
        panel_config: Optional panel of judges configuration
    """

    def __init__(self, model_name: str, 
                 llm_params: dict[str, Any],
                 panel_config: Optional[PanelOfJudgesConfig]):
        super().__init__(model_name, llm_params)

        # Initialize panel judges if enabled
        self.panel_config = panel_config
        self.panel_enabled = panel_config.enabled if panel_config else False
        self.judge_models: list[tuple[str, LiteLLMModel]] = []

        if self.panel_enabled and panel_config:
            self._initialize_panel_judges(panel_config, llm_params)
            print(
                f"✅ DeepEval LLM Manager: Panel mode with {len(self.judge_models)} judges"
            )
        else:
            print(f"✅ DeepEval LLM Manager: {self.model_name}")

    
    def _initialize_panel_judges(
        self, panel_config: PanelOfJudgesConfig, default_params: dict[str, Any]
    ) -> None:
        """Initialize all judge LLM instances.

        Args:
            panel_config: Panel of judges configuration
            default_params: Default LLM parameters to use when judge doesn't specify

        Raises:
            ValueError: If no judges are configured
        """
        # Validate that at least one judge is configured
        if not panel_config.judges or panel_config.judges is None:
            raise ValueError(
                "Panel of judges is enabled but no judges are configured. "
                "Please add at least one judge to the 'judges' list."
            )

        # Safety check: ensure judges is iterable
        judges_list = panel_config.judges if panel_config.judges else []
        if not judges_list:
            raise ValueError(
                "Panel of judges is enabled but no judges are configured. "
                "Please add at least one judge to the 'judges' list."
            )

        for judge_config in judges_list:
            # Validator should have auto-generated judge_id if not provided
            assert (
                judge_config.judge_id is not None
            ), "Judge ID must be set by validator"

            # Create LLMManager for this judge to handle provider-specific model names
            judge_llm_config = self._create_judge_llm_config(
                judge_config, default_params
            )
            judge_llm_manager = LLMManager(judge_llm_config)

            # Get properly formatted model name for this provider
            judge_model_name = judge_llm_manager.get_model_name()

            # Ensure we have valid int values with fallback defaults
            max_tokens: int = (
                judge_config.max_tokens or default_params.get("max_tokens") or 512
            )
            timeout: int = judge_config.timeout or default_params.get("timeout") or 300
            num_retries: int = judge_config.num_retries or default_params.get(
                "num_retries", 3
            )

            # Create DeepEval LLM model for this judge
            judge_llm = LiteLLMModel(
                model=judge_model_name,
                temperature=judge_config.temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                num_retries=num_retries,
            )

            self.judge_models.append((judge_config.judge_id, judge_llm))
            print(f"  → Judge {judge_config.judge_id}: {judge_model_name}")

    def _create_judge_llm_config(
        self, judge_config: JudgeConfig, default_params: dict[str, Any]
    ) -> Any:
        """Create LLMConfig from JudgeConfig.

        Args:
            judge_config: Judge configuration
            default_params: Default parameters to fill in missing values

        Returns:
            LLMConfig instance for this judge
        """
        # Ensure we have valid values with fallback defaults
        max_tokens: int = (
            judge_config.max_tokens or default_params.get("max_tokens") or 512
        )
        timeout: int = judge_config.timeout or default_params.get("timeout") or 300
        num_retries: int = judge_config.num_retries or default_params.get(
            "num_retries", 3
        )

        return LLMConfig(
            provider=judge_config.provider,
            model=judge_config.model,
            temperature=judge_config.temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            num_retries=num_retries,
            cache_dir=default_params.get("cache_dir", ".caches/llm_cache"),
            cache_enabled=default_params.get("cache_enabled", True),
        )

    def is_panel_enabled(self) -> bool:
        """Check if panel of judges mode is enabled.

        Returns:
            True if panel mode is enabled, False for single-LLM mode
        """
        return self.panel_enabled
 
    def get_llm(self) -> LiteLLMModel:
        """Get the primary DeepEval LLM model.

        In panel mode, returns the first judge's LLM.
        In single-LLM mode, returns the configured LLM.

        Returns:
            LiteLLMModel instance
        """
        if self.panel_enabled and self.judge_models:
            return self.judge_models[0][1]
        return self.llm_model
 
    def get_llms(self) -> list[tuple[str, LiteLLMModel]]:
        """Get all judge LLM models with their IDs.

        Returns:
            List of (judge_id, LiteLLMModel) tuples for panel mode,
            or single-item list for single-LLM mode
        """
        if self.panel_enabled and self.judge_models:
            return self.judge_models
        # Return primary LLM with a default judge ID for compatibility
        return [("primary", self.llm_model)]

    def get_panel_config(self) -> Optional[PanelOfJudgesConfig]:
        """Get the panel of judges configuration.

        Returns:
            PanelOfJudgesConfig if panel is enabled, None otherwise
        """
        return self.panel_config if self.panel_enabled else None

    def get_model_info(self) -> dict[str, Any]:

        """Get information about the configured model(s).

        Returns:
            Dictionary with model configuration details
        """
        base_info = {
             "model_name": self.model_name,
             "temperature": self.llm_params.get("temperature", 0.0),
             "max_tokens": self.llm_params.get("max_tokens"),
             "timeout": self.llm_params.get("timeout"),
             "num_retries": self.llm_params.get("num_retries", 3),
             "panel_enabled": self.panel_enabled,
         }

        if self.panel_enabled and self.judge_models:
            base_info["num_judges"] = len(self.judge_models)
            base_info["judges"] = [judge_id for judge_id, _ in self.judge_models]

        return base_info
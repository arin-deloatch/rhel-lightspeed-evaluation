"""Extended configuration loader for RHEL Lightspeed Evaluation.

Overrides API config parsing to support 'chat/completions' endpoint type.
"""

from typing import Any

from lightspeed_evaluation.core.models import (
    CoreConfig,
    EmbeddingConfig,
    LLMConfig,
    LoggingConfig,
    VisualizationConfig,
)
from lightspeed_evaluation.core.models.system import (
    JudgePanelConfig,
    LLMPoolConfig,
    QualityScoreConfig,
)
from lightspeed_evaluation.core.system import ConfigLoader

from rhel_lightspeed_evaluation.extensions.core.models.system import (
    APIConfigExt,
    SystemConfigExt,
)


class ConfigLoaderExt(ConfigLoader):
    """Extended configuration loader that uses APIConfigExt for chat/completions support."""

    system_config: SystemConfigExt | None

    def _create_system_config(self, config_data: dict[str, Any]) -> SystemConfigExt:
        """Create SystemConfigExt with APIConfigExt for chat/completions support."""
        metrics_metadata = config_data.get("metrics_metadata", {})
        quality_score_data = config_data.get("quality_score")

        turn_level_metadata = metrics_metadata.get("turn_level", {})
        conversation_level_metadata = metrics_metadata.get("conversation_level", {})

        quality_score_config = (
            QualityScoreConfig(**quality_score_data)
            if quality_score_data is not None
            else None
        )

        if quality_score_config is not None:
            self._process_quality_score_defaults(
                quality_score_config,
                turn_level_metadata,
                conversation_level_metadata,
            )

        llm_pool_data = config_data.get("llm_pool")
        llm_pool = LLMPoolConfig(**llm_pool_data) if llm_pool_data else None

        judge_panel_data = config_data.get("judge_panel")
        judge_panel = JudgePanelConfig(**judge_panel_data) if judge_panel_data else None

        storage_data = self._get_storage_config_with_backward_compat(config_data)
        storage_backends = self._parse_storage_config(storage_data)

        return SystemConfigExt(
            core=CoreConfig(**config_data.get("core", {})),
            llm=LLMConfig(**config_data.get("llm", {})),
            embedding=EmbeddingConfig(**config_data.get("embedding") or {}),
            api=APIConfigExt(**config_data.get("api", {})),
            agents=config_data.get("agents"),
            storage=storage_backends,
            logging=LoggingConfig(**config_data.get("logging", {})),
            visualization=VisualizationConfig(**config_data.get("visualization", {})),
            llm_pool=llm_pool,
            judge_panel=judge_panel,
            quality_score=quality_score_config,
            default_turn_metrics_metadata=turn_level_metadata,
            default_conversation_metrics_metadata=conversation_level_metadata,
        )

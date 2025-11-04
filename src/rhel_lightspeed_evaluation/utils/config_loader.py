"""Extended configuration loader for RHEL Lightspeed Evaluation.

This module provides an extended ConfigLoader that loads the extended
SystemConfig with additional fields (geval, panel, etc.) that are specific
to the RHEL Lightspeed evaluation extensions.
"""

from typing import Any

from lightspeed_evaluation.core.system import ConfigLoader

from rhel_lightspeed_evaluation.models.system_ext import (
    CoreConfig,
    EmbeddingConfig,
    GEvalConfig,
    LLMConfig,
    LoggingConfig,
    OutputConfig,
    PanelConfig,
    SystemConfig,
    VisualizationConfig,
)


class ExtendedConfigLoader(ConfigLoader):
    """Extended configuration loader that uses extended SystemConfig.

    This loader extends the base ConfigLoader to support additional
    configuration sections specific to RHEL Lightspeed evaluation:
    - geval: GEval metrics configuration
    - panel: Panel of judges configuration
    """

    def _create_system_config(self, config_data: dict[str, Any]) -> SystemConfig:
        """Create extended SystemConfig object from validated configuration data.

        This method overrides the base implementation to include additional
        configuration sections that are specific to RHEL Lightspeed evaluation.

        Args:
            config_data: Raw configuration dictionary from YAML

        Returns:
            Extended SystemConfig instance with all configuration sections
        """
        metrics_metadata = config_data.get("metrics_metadata", {})

        # Get API config from base package
        from lightspeed_evaluation.core.models import APIConfig

        return SystemConfig(
            core=CoreConfig(**config_data.get("core", {})),
            llm=LLMConfig(**config_data.get("llm", {})),
            embedding=EmbeddingConfig(**config_data.get("embedding") or {}),
            api=APIConfig(**config_data.get("api", {})),
            output=OutputConfig(**config_data.get("output", {})),
            logging=LoggingConfig(**config_data.get("logging", {})),
            visualization=VisualizationConfig(**config_data.get("visualization", {})),
            panel=PanelConfig(**config_data.get("panel", {})),
            geval=GEvalConfig(**config_data.get("geval", {})),
            default_turn_metrics_metadata=metrics_metadata.get("turn_level", {}),
            default_conversation_metrics_metadata=metrics_metadata.get(
                "conversation_level", {}
            ),
        )

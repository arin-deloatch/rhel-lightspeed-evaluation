"""Extended Evaluation Pipeline with chat/completions API client support."""

import logging

from lightspeed_evaluation.core.api import APIClient
from lightspeed_evaluation.pipeline.evaluation.pipeline import EvaluationPipeline

from rhel_lightspeed_evaluation.extensions.core.api import APIClientExt

logger = logging.getLogger(__name__)


class EvaluationPipelineExt(EvaluationPipeline):
    """Evaluation pipeline that uses APIClientExt for chat/completions endpoints."""

    def _create_api_client(self) -> APIClientExt | APIClient | None:
        """Create API client, using APIClientExt for chat/completions endpoints."""
        config = self.config_loader.system_config
        if config is None:
            raise ValueError("SystemConfig must be loaded before creating API client")
        if not config.api.enabled:
            return None

        api_config = config.api
        logger.info("Setting up API client: %s", api_config.api_base)

        if api_config.endpoint_type in ("chat/completions", "infer"):
            client = APIClientExt(api_config)
        else:
            client = APIClient(api_config)

        logger.info("API client initialized for %s endpoint", api_config.endpoint_type)
        return client

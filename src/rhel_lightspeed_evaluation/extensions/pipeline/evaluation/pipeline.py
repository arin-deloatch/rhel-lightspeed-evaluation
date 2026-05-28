"""Extended Evaluation Pipeline with API client support.

Uses APIClientExt for all endpoint types. APIClientExt adds chat/completions
support and fixes the duplicate /api/lightspeed/ path in the base class's
infer endpoint handler.
"""

import logging

from lightspeed_evaluation.pipeline.evaluation.pipeline import EvaluationPipeline

from rhel_lightspeed_evaluation.extensions.core.api import APIClientExt

logger = logging.getLogger(__name__)


class EvaluationPipelineExt(EvaluationPipeline):
    """Evaluation pipeline that uses APIClientExt for all endpoint types."""

    def _create_api_client(self) -> APIClientExt | None:
        """Create API client using APIClientExt."""
        config = self.config_loader.system_config
        if config is None:
            raise ValueError("SystemConfig must be loaded before creating API client")
        if not config.api.enabled:
            return None

        api_config = config.api
        logger.info("Setting up API client: %s", api_config.api_base)

        client = APIClientExt(api_config)

        logger.info("API client initialized for %s endpoint", api_config.endpoint_type)
        return client

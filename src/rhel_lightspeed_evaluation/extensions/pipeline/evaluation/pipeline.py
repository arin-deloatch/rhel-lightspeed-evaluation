"""RHEL HTTP driver integration with the upstream evaluation pipeline."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from lightspeed_evaluation.core.models import EvaluationData, EvaluationResult
from lightspeed_evaluation.core.models.agents import HttpApiAgentConfig
from lightspeed_evaluation.core.models.data import DatasetMetadata
from lightspeed_evaluation.pipeline.evaluation.driver import AgentDriver, HttpApiDriver
from lightspeed_evaluation.pipeline.evaluation.pipeline import EvaluationPipeline
from lightspeed_evaluation.pipeline.evaluation.registry import AGENT_DRIVERS, AgentDriverRegistry

from rhel_lightspeed_evaluation.extensions.core.api import APIClientExt
from rhel_lightspeed_evaluation.extensions.core.models.system import HttpApiAgentConfigExt
from rhel_lightspeed_evaluation.extensions.pipeline.evaluation.checkpoint import (
    CheckpointStore,
    validate_storage,
)
from rhel_lightspeed_evaluation.extensions.pipeline.evaluation.processor import (
    ConversationProcessorExt,
)

logger = logging.getLogger(__name__)


class HttpApiDriverExt(HttpApiDriver):
    """Keep RHEL endpoint support for default and per-conversation agents."""

    def validate_config(self, config: dict[str, Any]) -> HttpApiAgentConfigExt:
        return HttpApiAgentConfigExt.model_validate(config)

    def _create_api_client(self, config: HttpApiAgentConfig) -> APIClientExt:
        return APIClientExt(config)


class EvaluationPipelineExt(EvaluationPipeline):
    """Use a private registry so other upstream consumers are unaffected."""

    conversation_processor: ConversationProcessorExt
    _checkpoint: CheckpointStore | None = None

    def run_evaluation(
        self,
        evaluation_data: list[EvaluationData],
        original_data_path: str | None = None,
        dataset_metadata: DatasetMetadata | None = None,
        *,
        resume: bool = False,
    ) -> list[EvaluationResult]:
        directory = Path(self.output_dir) / ".rhel-checkpoint" / "conversations"
        if not resume:
            if directory.exists():
                raise ValueError("Output contains checkpoints; use --resume or a fresh directory")
            return super().run_evaluation(evaluation_data, original_data_path, dataset_metadata)
        validate_storage(self.system_config)
        with CheckpointStore(directory) as checkpoint:
            checkpoint.validate(self.system_config, evaluation_data, dataset_metadata)
            self._checkpoint = checkpoint
            try:
                return super().run_evaluation(evaluation_data, original_data_path, dataset_metadata)
            finally:
                self._checkpoint = None

    def _process_eval_data(self, evaluation_data: list[EvaluationData]) -> list[EvaluationResult]:
        if self._checkpoint is None:
            return super()._process_eval_data(evaluation_data)
        saved = self._checkpoint.load(evaluation_data)
        results_by_id = {identifier: record.results for identifier, record in saved.items()}
        pending = []
        for index, conversation in enumerate(evaluation_data):
            record = saved.get(conversation.conversation_group_id)
            if record is None:
                pending.append(conversation)
            else:
                evaluation_data[index] = record.conversation
                self.storage_backend.save_run(record.results)
        logger.info("Resume: %d completed conversations, %d remaining", len(saved), len(pending))
        failure: Exception | None = None
        with ThreadPoolExecutor(max_workers=self.system_config.core.max_threads) as executor:
            futures = {executor.submit(self._process_conversation, item): item for item in pending}
            for future in as_completed(futures):
                conversation = futures[future]
                try:
                    results = future.result()
                except Exception as error:
                    # Drain other completed conversations so one failed worker does not lose them.
                    failure = failure or error
                    continue
                self._checkpoint.save(conversation, results)
                self.storage_backend.save_run(results)
                results_by_id[conversation.conversation_group_id] = results
        if failure is not None:
            raise failure
        return [
            result
            for conversation in evaluation_data
            for result in results_by_id[conversation.conversation_group_id]
        ]

    def _initialize_components(self) -> None:
        super()._initialize_components()
        self.conversation_processor = ConversationProcessorExt(
            self.config_loader, self.conversation_processor.components
        )

    def _create_default_driver(self) -> AgentDriver:
        self._registry = AgentDriverRegistry({**AGENT_DRIVERS, "http_api": HttpApiDriverExt})
        return super()._create_default_driver()

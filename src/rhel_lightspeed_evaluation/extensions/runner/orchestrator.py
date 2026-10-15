"""Use upstream run planning/reporting with RHEL workers.

Upstream does not yet accept a pipeline or config factory. Keep the worker
boundary here so spawned processes use the same extensions as sequential runs.
"""

import logging
import multiprocessing
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rhel_lightspeed_evaluation.extensions.core.llm.patches import apply_patches

if TYPE_CHECKING:
    from lightspeed_evaluation.core.models import EvaluationData
    from lightspeed_evaluation.pipeline.behavioral.models import RunContext, RunResult

    from rhel_lightspeed_evaluation.extensions.core.models.system import SystemConfigExt

logger = logging.getLogger(__name__)


def run(
    config: "SystemConfigExt",
    data: "list[EvaluationData]",
    output_dir: str,
    **extra: Any,
) -> "list[RunResult]":
    """Plan the upstream agent/repeat matrix and run it with RHEL extensions."""
    apply_patches()
    from lightspeed_evaluation.pipeline.behavioral import orchestrator as upstream

    agents = config.agents
    if agents is None or not agents.default.agent:
        return []
    names = upstream._build_agent_set(agents.default.agent, data)
    resume = extra.get("resume", False)
    timestamp = "resume" if resume else datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    contexts = upstream._build_run_contexts(
        config,
        data,
        {
            **extra,
            "default_agents": agents.default.agent,
            "run_matrix": [
                (name, i) for name in names for i in range(1, agents.default.repeat + 1)
            ],
            "output_base": output_dir,
            "timestamp": timestamp,
        },
    )
    for context in contexts:
        context.extra = {**(context.extra or {}), "resume": resume}
    if agents.default.parallel and len(contexts) > 1:
        upstream._warn_resource_usage(len(contexts), config)
        with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count() - 1)) as executor:
            results = list(executor.map(_run_single, contexts))
    else:
        results = [_run_single(ctx) for ctx in contexts]
    return results


def _run_single(ctx: "RunContext") -> "RunResult":
    """Initialize judge patches inside the worker before loading heavy modules."""
    apply_patches()
    from lightspeed_evaluation.core.models import EvaluationData
    from lightspeed_evaluation.core.models.data import DatasetMetadata
    from lightspeed_evaluation.core.system.exceptions import ConfigurationError, DataValidationError
    from lightspeed_evaluation.pipeline.behavioral import orchestrator as upstream
    from lightspeed_evaluation.pipeline.behavioral.models import RunResult

    from rhel_lightspeed_evaluation.extensions.core.models.system import SystemConfigExt
    from rhel_lightspeed_evaluation.extensions.core.system import ConfigLoaderExt
    from rhel_lightspeed_evaluation.extensions.pipeline.evaluation import EvaluationPipelineExt

    Path(ctx.run_output_dir).mkdir(parents=True, exist_ok=True)
    try:
        config = SystemConfigExt.model_validate(
            upstream._clone_config_for_run(
                ctx.config_dict,
                ctx.agent_name,
                disable_cache=ctx.run_index > 1,
                run_output_dir=ctx.run_output_dir,
            )
        )
        conversations = [EvaluationData.model_validate(d) for d in ctx.eval_data_dicts]
        filtered = upstream._filter_conversations(conversations, ctx.agent_name, ctx.default_agents)
        if not filtered:
            return RunResult(
                agent_name=ctx.agent_name,
                run_index=ctx.run_index,
                output_dir=ctx.run_output_dir,
                success=True,
                summary=upstream._empty_summary(),
            )
        data = upstream._pin_conversations_to_agent(filtered, ctx.agent_name)
        extra = ctx.extra or {}
        metadata = extra.get("dataset_metadata_dict")
        pipeline = EvaluationPipelineExt(ConfigLoaderExt.from_config(config), ctx.run_output_dir)
        try:
            results = pipeline.run_evaluation(
                data,
                original_data_path=extra.get("original_data_path"),
                dataset_metadata=DatasetMetadata.model_validate(metadata) if metadata else None,
                resume=extra.get("resume", False),
            )
        finally:
            pipeline.close()
        return RunResult(
            agent_name=ctx.agent_name,
            run_index=ctx.run_index,
            output_dir=ctx.run_output_dir,
            success=True,
            summary=upstream._make_summary(results),
        )
    except (ConfigurationError, DataValidationError):
        raise
    except Exception:
        error = traceback.format_exc()
        logger.error("Run %s/%s failed: %s", ctx.agent_name, ctx.run_index, error)
        return RunResult(
            agent_name=ctx.agent_name,
            run_index=ctx.run_index,
            output_dir=ctx.run_output_dir,
            success=False,
            error=error,
        )

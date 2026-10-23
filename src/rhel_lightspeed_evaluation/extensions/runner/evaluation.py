"""RHEL evaluation runner using upstream agent planning and result statistics."""

import argparse
import shutil
import sys
import traceback
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from lightspeed_evaluation.core.storage import get_file_config
from lightspeed_evaluation.core.system.exceptions import (
    ConfigurationError,
    DataValidationError,
    StorageError,
)
from lightspeed_evaluation.runner.evaluation import (
    _aggregate_totals,
    _clear_caches,
    _copy_flat_output,
    _print_run_summary,
    create_eval_parser,
)

from rhel_lightspeed_evaluation.extensions.core.llm.patches import apply_patches
from rhel_lightspeed_evaluation.extensions.core.system import ConfigLoaderExt


def _evaluate(config: Any, data: Any, output_dir: str | None = None, **kwargs: Any) -> Any:
    """Run the offline pipeline with RHEL extensions and close its resources."""
    from rhel_lightspeed_evaluation.extensions.pipeline.evaluation import EvaluationPipelineExt

    pipeline = EvaluationPipelineExt(ConfigLoaderExt.from_config(config), output_dir)
    try:
        return pipeline.run_evaluation(data, **kwargs)
    finally:
        pipeline.close()


def run_evaluation(eval_args: argparse.Namespace) -> dict[str, int] | None:
    """Keep the checkpoint lock until all workers and reports have finished."""
    with ExitStack() as scope:
        return _run_evaluation(eval_args, scope)


def _run_evaluation(  # pylint: disable=too-many-locals
    eval_args: argparse.Namespace,
    scope: ExitStack,
) -> dict[str, int] | None:
    """Run the complete evaluation pipeline.

    Args:
        eval_args: Parsed command line arguments

    Returns:
        dict: Summary statistics with keys TOTAL, PASS, FAIL, ERROR, SKIPPED.
    """
    print("🚀 Lightspeed Evaluation Framework")
    print("=" * 50)

    try:
        resume = getattr(eval_args, "resume", False)
        if resume and not eval_args.output_dir:
            raise ValueError("--resume requires an explicit --output-dir")
        if resume and eval_args.cache_warmup:
            raise ValueError("--resume cannot be combined with --cache-warmup")
        print("🔧 Loading Configuration & Setting up environment...")
        loader = ConfigLoaderExt()
        system_config = loader.load_system_config(eval_args.system_config)

        # Clear caches if cache warmup mode is enabled
        if eval_args.cache_warmup:
            print("\n🔥 Cache warmup mode: Clearing existing caches...")
            _clear_caches(system_config)

        apply_patches()

        # Import heavy modules after environment is configured
        print("\n📋 Loading Heavy Modules...")
        # pylint: disable=import-outside-toplevel
        from lightspeed_evaluation.core.output import OutputHandler
        from lightspeed_evaluation.core.output.statistics import compute_overall_stats
        from lightspeed_evaluation.core.storage import FileBackendConfig
        from lightspeed_evaluation.core.system import DataValidator

        from rhel_lightspeed_evaluation.extensions.pipeline.evaluation.checkpoint import (
            CheckpointStore,
            validate_storage,
        )
        from rhel_lightspeed_evaluation.extensions.runner.orchestrator import (
            run as orchestrator_run,
        )

        # pylint: enable=import-outside-toplevel
        print("✅ Configuration loaded & Setup is done !")

        # Load, filter, and validate evaluation data
        data_validator = DataValidator(
            api_enabled=system_config.agents is not None and system_config.agents.enabled,
            fail_on_invalid_data=system_config.core.fail_on_invalid_data,
            system_config=system_config,
        )
        evaluation_data = data_validator.load_evaluation_data(
            eval_args.eval_data,
            tags=eval_args.tags,
            conv_ids=eval_args.conv_ids,
            metrics=eval_args.metrics,
        )
        dataset_metadata = data_validator.dataset_metadata

        output_dir = eval_args.output_dir or get_file_config(system_config.storage).output_dir
        checkpoint_dir = Path(output_dir) / ".rhel-checkpoint"
        if (Path(output_dir) / ".completed_ids").exists():
            raise ValueError(
                "Legacy fork checkpoints are unsupported; use a fresh output directory"
            )
        if resume:
            validate_storage(system_config)
            checkpoint = scope.enter_context(CheckpointStore(checkpoint_dir))
            checkpoint.validate(system_config, evaluation_data, dataset_metadata)
        elif checkpoint_dir.exists():
            raise ValueError("Output contains checkpoints; use --resume or a fresh directory")

        print(f"✅ System config: {system_config.llm.provider}/{system_config.llm.model}")

        # Handle case where no conversations match the filter
        if len(evaluation_data) == 0:
            print("\n⚠️ No conversation groups matched the filter criteria")
            print("   Nothing to evaluate - returning empty results")
            return {"TOTAL": 0, "PASS": 0, "FAIL": 0, "ERROR": 0, "SKIPPED": 0}

        # Run evaluation
        print("\n🔄 Running Evaluation...")
        has_agents = (
            system_config.agents is not None
            and system_config.agents.enabled
            and system_config.agents.default.agent
        )

        if not has_agents:
            # Offline mode: run pipeline directly (no agents to orchestrate)
            results = _evaluate(
                system_config,
                evaluation_data,
                output_dir=eval_args.output_dir,
                original_data_path=eval_args.eval_data,
                dataset_metadata=dataset_metadata,
                resume=resume,
            )
            file_entries = [c for c in system_config.storage if isinstance(c, FileBackendConfig)]
            if not file_entries:
                file_config = get_file_config(system_config.storage)
                handler = OutputHandler(
                    output_dir=eval_args.output_dir or file_config.output_dir,
                    base_filename=file_config.base_filename,
                    system_config=system_config,
                    file_config=file_config,
                )
                handler.generate_reports(results, evaluation_data)
            summary = compute_overall_stats(results)
            out_dir = eval_args.output_dir or get_file_config(system_config.storage).output_dir
            totals: dict[str, int] = {
                "TOTAL": summary.total,
                "PASS": summary.passed,
                "FAIL": summary.failed,
                "ERROR": summary.error,
                "SKIPPED": summary.skipped,
            }
            print("\n🎉 Evaluation Complete!")
            _print_run_summary(totals, output_dir=out_dir)
            return totals

        # Agent mode: run via orchestrator
        output_dir = eval_args.output_dir or get_file_config(system_config.storage).output_dir
        run_results = orchestrator_run(
            system_config,
            evaluation_data,
            output_dir,
            original_data_path=eval_args.eval_data,
            dataset_metadata_dict=(dataset_metadata.model_dump() if dataset_metadata else None),
            resume=resume,
        )
        if any(not result.success for result in run_results):
            return None
        totals = _aggregate_totals(run_results)
        if resume and len(run_results) == 1:
            # Upstream copies the whole tree, including locks and per-run checkpoints.
            shutil.copytree(
                run_results[0].output_dir,
                output_dir,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(".rhel-checkpoint"),
            )
        elif not resume:
            _copy_flat_output(run_results, output_dir)

        print("\n🎉 Evaluation Complete!")
        _print_run_summary(totals, run_results=run_results)

        return totals

    except (
        OSError,
        ValueError,
        RuntimeError,
        ConfigurationError,
        DataValidationError,
        StorageError,
    ) as e:
        print(f"\n❌ Evaluation failed: {e}")
        traceback.print_exc()
        return None


def main() -> int:
    """Run the RHEL CLI with upstream options and opt-in durable resume."""
    parser = create_eval_parser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Save conversation checkpoints and resume matching work in --output-dir",
    )
    args = parser.parse_args()
    return 0 if run_evaluation(args) is not None else 1


if __name__ == "__main__":
    sys.exit(main())

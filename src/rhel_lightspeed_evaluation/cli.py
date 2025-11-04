"""Custom runner that integrates RHEL Lightspeed extensions with the framework.

This runner extends the lightspeed-evaluation framework to support:
- GEval metrics with runtime-defined criteria
- Panel of Judges for multi-model evaluation (future)
"""

import argparse
import sys
import traceback
from pathlib import Path
from typing import Optional

from lightspeed_evaluation.core.llm.manager import LLMManager
from lightspeed_evaluation.core.output import OutputHandler
from lightspeed_evaluation.core.output.statistics import calculate_basic_stats
from lightspeed_evaluation.core.system import DataValidator
from lightspeed_evaluation.pipeline.evaluation import EvaluationPipeline

from rhel_lightspeed_evaluation.extensions.geval import GEvalMetrics
from rhel_lightspeed_evaluation.utils import ExtendedConfigLoader


class ExtendedEvaluationPipeline(EvaluationPipeline):
    """Extended pipeline that adds GEval metric support.

    This pipeline extends the base EvaluationPipeline to register
    our custom GEval handler with the metrics evaluator and auto-apply
    default GEval metrics from system configuration.
    """

    def _initialize_components(self) -> None:
        """Initialize components and add GEval handler."""
        # Call parent to initialize all standard components
        super()._initialize_components()

        # Get the metrics evaluator created by parent
        metrics_evaluator = self.conversation_processor.components.metrics_evaluator

        # Create LLM manager for GEval
        llm_manager = LLMManager.from_system_config(self.system_config)

        # Initialize GEval handler with config from system.yaml
        geval_handler = GEvalMetrics(
            llm_manager=llm_manager,
            registry_path=self.system_config.geval.registry_path,
        )

        # Register GEval handler in the metrics evaluator
        metrics_evaluator.handlers["geval"] = geval_handler

    def run_evaluation(
        self,
        evaluation_data: list,
        original_data_path: Optional[str] = None,
    ) -> list:
        """Run evaluation with auto-applied default GEval metrics.

        This method extends the parent run_evaluation to automatically
        inject default GEval metrics from system configuration before
        processing evaluations.

        Args:
            evaluation_data: List of conversation data to evaluate
            original_data_path: Path to original data file for saving updates

        Returns:
            List of evaluation results
        """
        # Auto-apply default GEval metrics from system config
        self._apply_default_geval_metrics(evaluation_data)

        # Call parent implementation
        return super().run_evaluation(evaluation_data, original_data_path)

    def _apply_default_geval_metrics(self, evaluation_data: list) -> None:
        """Apply default GEval metrics to evaluation data.

        This method injects default turn and conversation metrics from
        system configuration into evaluation data that doesn't already
        specify metrics.

        Args:
            evaluation_data: List of EvaluationData objects to modify
        """
        geval_config = self.system_config.geval

        # Only proceed if GEval is enabled and defaults are configured
        if not geval_config.enabled:
            return

        default_turn = geval_config.default_turn_metrics
        default_conv = geval_config.default_conversation_metrics

        # Nothing to do if no defaults configured
        if not default_turn and not default_conv:
            return

        # Process each conversation
        for conv in evaluation_data:
            # Apply default conversation metrics (prepend to existing)
            if default_conv:
                if conv.conversation_metrics is None:
                    conv.conversation_metrics = default_conv.copy()
                else:
                    # Prepend defaults, avoiding duplicates
                    existing = set(conv.conversation_metrics)
                    new_metrics = [m for m in default_conv if m not in existing]
                    conv.conversation_metrics = new_metrics + conv.conversation_metrics

            # Apply default turn metrics (prepend to existing)
            if default_turn:
                for turn in conv.turns:
                    if turn.turn_metrics is None:
                        turn.turn_metrics = default_turn.copy()
                    else:
                        # Prepend defaults, avoiding duplicates
                        existing = set(turn.turn_metrics)
                        new_metrics = [m for m in default_turn if m not in existing]
                        turn.turn_metrics = new_metrics + turn.turn_metrics


def run_evaluation(
    system_config_path: str,
    evaluation_data_path: str,
    output_dir: Optional[str] = None,
) -> Optional[dict[str, int]]:
    """Run evaluation with RHEL Lightspeed extensions.

    This follows the same pattern as the framework's runner but uses
    our extended pipeline to support GEval metrics.

    Args:
        system_config_path: Path to system.yaml
        evaluation_data_path: Path to evaluation_data.yaml
        output_dir: Optional override for output directory

    Returns:
        dict: Summary statistics with keys TOTAL, PASS, FAIL, ERROR
    """
    print("🚀 RHEL Lightspeed Evaluation Framework")
    print("=" * 50)

    try:
        # Step 0: Setup environment from config
        print("🔧 Loading Configuration...")
        loader = ExtendedConfigLoader()
        system_config = loader.load_system_config(system_config_path)

        print("✅ Configuration loaded")

        llm_config = system_config.llm
        output_config = system_config.output

        # Step 1: Load and validate evaluation data
        data_validator = DataValidator(api_enabled=system_config.api.enabled)
        evaluation_data = data_validator.load_evaluation_data(evaluation_data_path)

        print(f"✅ System config: {llm_config.provider}/{llm_config.model}")
        print(f"✅ Evaluation data: {len(evaluation_data)} conversation groups")

        # Step 2: Run evaluation with extended pipeline
        print("\n⚙️ Initializing Extended Evaluation Pipeline (with GEval support)...")
        pipeline = ExtendedEvaluationPipeline(loader, output_dir)

        print("\n🔄 Running Evaluation...")
        try:
            results = pipeline.run_evaluation(evaluation_data, evaluation_data_path)
        finally:
            pipeline.close()

        # Step 3: Generate reports and calculate stats
        print("\n📊 Generating Reports...")
        output_handler = OutputHandler(
            output_dir=output_dir or output_config.output_dir,
            base_filename=output_config.base_filename,
            system_config=system_config,
        )

        # Generate reports based on configuration
        output_handler.generate_reports(results)

        print("\n🎉 Evaluation Complete!")
        print(f"📊 {len(results)} evaluations completed")
        print(f"📁 Reports generated in: {output_handler.output_dir}")

        # Step 4: Final Summary
        summary = calculate_basic_stats(results)
        print(
            f"✅ Pass: {summary['PASS']}, ❌ Fail: {summary['FAIL']}, ⚠️ Error: {summary['ERROR']}"
        )
        if summary["ERROR"] > 0:
            print(f"⚠️ {summary['ERROR']} evaluations had errors - check detailed report")

        return {
            "TOTAL": summary["TOTAL"],
            "PASS": summary["PASS"],
            "FAIL": summary["FAIL"],
            "ERROR": summary["ERROR"],
        }

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n❌ Evaluation failed: {e}")
        traceback.print_exc()
        return None


def main() -> int:
    """Main entry point for extended evaluation runner.

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="RHEL Lightspeed Evaluation with GEval support"
    )
    parser.add_argument(
        "--system-config",
        default="config/system.yaml",
        help="Path to system configuration file (default: config/system.yaml)",
    )
    parser.add_argument(
        "--eval-data",
        default="config/evaluation_data.yaml",
        help="Path to evaluation data file (default: config/evaluation_data.yaml)",
    )
    parser.add_argument("--output-dir", help="Override output directory (optional)")

    args = parser.parse_args()

    # Validate paths
    if not Path(args.system_config).exists():
        print(f"Error: System config not found: {args.system_config}", file=sys.stderr)
        return 1

    if not Path(args.eval_data).exists():
        print(f"Error: Evaluation data not found: {args.eval_data}", file=sys.stderr)
        return 1

    # Run evaluation
    summary = run_evaluation(args.system_config, args.eval_data, args.output_dir)

    return 0 if summary is not None else 1


if __name__ == "__main__":
    sys.exit(main())

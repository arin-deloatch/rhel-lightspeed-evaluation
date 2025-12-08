from typing import Any
from pathlib import Path
from datetime import datetime
import json

from lightspeed_evaluation.core.constants import DEFAULT_OUTPUT_DIR
from lightspeed_evaluation.core.output.generator import OutputHandler
from rhel_lightspeed_evaluation.extensions.core.models.data import EvaluationResultExt



class OutputHandlerExt(OutputHandler):
    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR, 
                 base_filename: str = "evaluation", 
                 system_config: Any | None = None) -> None:
        super().__init__(output_dir, base_filename, system_config)

        """Initialize Output handler."""
        self.output_dir = Path(output_dir)
        self.base_filename = base_filename
        self.system_config = system_config
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"✅ Output handler initialized: {self.output_dir}")

    # Adding support for judge identification
    def _generate_json_summary(
        self,
        results: list[EvaluationResultExt],
        base_filename: str,
        basic_stats: dict[str, Any],
        detailed_stats: dict[str, Any],
    ) -> Path:
        """Generate JSON summary report."""
        json_file = self.output_dir / f"{base_filename}_summary.json"

        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_evaluations": len(results),
            "summary_stats": {
                "overall": basic_stats,
                "by_metric": detailed_stats["by_metric"],
                "by_conversation": detailed_stats["by_conversation"],
            },
            "results": [
                {
                    "conversation_group_id": r.conversation_group_id,
                    "turn_id": r.turn_id,
                    "metric_identifier": r.metric_identifier,
                    "judge_id": r.judge_id,
                    "result": r.result,
                    "score": r.score,
                    "threshold": r.threshold,
                    "execution_time": round(r.execution_time, 3),
                }
                for r in results
            ],
        }

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return json_file
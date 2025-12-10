from typing import Any
from pathlib import Path
from datetime import datetime
import json

from pydantic import BaseModel

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

    def _get_included_config_sections(self) -> list[str]:
        """Get list of configuration sections to include in summaries."""
        if self.system_config is not None and hasattr(self.system_config, 'output'):
            if hasattr(self.system_config.output, 'summary_config_sections'):
                return self.system_config.output.summary_config_sections
        # Default sections if not configured
        return ["llm", "embedding", "api", "panel_of_judges"]

    def _format_field_value(
        self,
        display_name: str,
        field_value: Any,
        prefix: str,
        indent: int,
        skip_fields: set[str]
    ) -> list[str]:
        """Format a single field value, handling nested structures."""
        lines = []

        if isinstance(field_value, (dict, list)) and field_value:
            lines.append(f"{prefix}{display_name}:")
            nested_lines = self._format_config_section(
                field_value,
                indent=indent + 2,
                skip_fields=skip_fields
            )
            lines.extend(nested_lines)
        elif isinstance(field_value, bool):
            lines.append(f"{prefix}{display_name}: {field_value}")
        elif field_value is not None:
            lines.append(f"{prefix}{display_name}: {field_value}")

        return lines

    def _format_basemodel(
        self,
        config: BaseModel,
        prefix: str,
        indent: int,
        skip_fields: set[str]
    ) -> list[str]:
        """Format a Pydantic BaseModel."""
        lines = []
        for field_name, field_value in config.model_dump().items():
            if field_name not in skip_fields:
                display_name = field_name.replace("_", " ").title()
                lines.extend(self._format_field_value(
                    display_name, field_value, prefix, indent, skip_fields
                ))
        return lines

    def _format_dict(
        self,
        config: dict,
        prefix: str,
        indent: int,
        skip_fields: set[str]
    ) -> list[str]:
        """Format a dictionary."""
        lines = []
        for key, value in config.items():
            if key not in skip_fields:
                display_name = str(key).replace("_", " ").title()
                lines.extend(self._format_field_value(
                    display_name, value, prefix, indent, skip_fields
                ))
        return lines

    def _get_list_item_label(self, item: dict | BaseModel, idx: int, prefix: str) -> str:
        """Generate a label for a list item."""
        item_id = None
        if isinstance(item, BaseModel) and hasattr(item, 'judge_id'):
            item_id = getattr(item, 'judge_id')
        elif isinstance(item, dict) and 'judge_id' in item:
            item_id = item['judge_id']

        if item_id:
            return f"{prefix}Judge {idx} ({item_id}):"
        return f"{prefix}Item {idx}:"

    def _format_list(
        self,
        config: list,
        prefix: str,
        indent: int,
        skip_fields: set[str]
    ) -> list[str]:
        """Format a list."""
        lines = []
        for idx, item in enumerate(config, 1):
            if isinstance(item, (dict, BaseModel)):
                lines.append(self._get_list_item_label(item, idx, prefix))
                nested_lines = self._format_config_section(
                    item,
                    indent=indent + 2,
                    skip_fields=skip_fields
                )
                lines.extend(nested_lines)
            else:
                lines.append(f"{prefix}- {item}")
        return lines

    def _format_config_section(
        self,
        config: BaseModel | dict | list | Any,
        indent: int = 2,
        skip_fields: set[str] | None = None
    ) -> list[str]:
        """
        Recursively format configuration parameters for text output.

        Args:
            config: Configuration object (Pydantic model, dict, list, or primitive)
            indent: Number of spaces for indentation
            skip_fields: Set of field names to skip (e.g., sensitive data)

        Returns:
            List of formatted strings
        """
        if skip_fields is None:
            skip_fields = {"cache_dir"}  # Skip verbose fields by default

        prefix = " " * indent

        if isinstance(config, BaseModel):
            return self._format_basemodel(config, prefix, indent, skip_fields)
        elif isinstance(config, dict):
            return self._format_dict(config, prefix, indent, skip_fields)
        elif isinstance(config, list):
            return self._format_list(config, prefix, indent, skip_fields)

        return []

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
    
    def _generate_text_summary(
        self,
        results: list[EvaluationResultExt],
        base_filename: str,
        basic_stats: dict[str, Any],
        detailed_stats: dict[str, Any],
    ) -> Path:
        """Generate human-readable text summary."""
        txt_file = self.output_dir / f"{base_filename}_summary.txt"

        stats = {
            "overall": basic_stats,
            "by_metric": detailed_stats["by_metric"],
            "by_conversation": detailed_stats["by_conversation"],
        }

        with open(txt_file, "w", encoding="utf-8") as f:
            f.write("LSC Evaluation Framework - Summary Report\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Evaluations: {len(results)}\n\n")

            # Overall statistics
            f.write("Overall Statistics:\n")
            f.write("-" * 20 + "\n")
            f.write(
                f"Pass: {stats['overall']['PASS']} ({stats['overall']['pass_rate']:.1f}%)\n"
            )
            f.write(
                f"Fail: {stats['overall']['FAIL']} ({stats['overall']['fail_rate']:.1f}%)\n"
            )
            f.write(
                f"Error: {stats['overall']['ERROR']} ({stats['overall']['error_rate']:.1f}%)\n\n"
            )

            # By metric breakdown
            if stats["by_metric"]:
                f.write("By Metric:\n")
                f.write("-" * 10 + "\n")
                for metric, metric_stats in stats["by_metric"].items():
                    f.write(f"{metric}:\n")
                    f.write(
                        f"  Pass: {metric_stats['pass']} ({metric_stats['pass_rate']:.1f}%)\n"
                    )
                    f.write(
                        f"  Fail: {metric_stats['fail']} ({metric_stats['fail_rate']:.1f}%)\n"
                    )
                    f.write(
                        f"  Error: {metric_stats['error']} ({metric_stats['error_rate']:.1f}%)\n"
                    )
                    if (
                        "score_statistics" in metric_stats
                        and metric_stats["score_statistics"]["count"] > 0
                    ):
                        score_stats = metric_stats["score_statistics"]
                        f.write("  Score Statistics:\n")
                        f.write(f"    Mean: {score_stats['mean']:.3f}\n")
                        f.write(f"    Median: {score_stats['median']:.3f}\n")
                        f.write(
                            f"    Min: {score_stats['min']:.3f}, Max: {score_stats['max']:.3f}\n"
                        )
                        if score_stats["count"] > 1:
                            f.write(f"    Std Dev: {score_stats['std']:.3f}\n")
                    f.write("\n")

            # By conversation breakdown
            if stats["by_conversation"]:
                f.write("By Conversation:\n")
                f.write("-" * 15 + "\n")
                for conv_id, conv_stats in stats["by_conversation"].items():
                    f.write(f"{conv_id}:\n")
                    f.write(
                        f"  Pass: {conv_stats['pass']} ({conv_stats['pass_rate']:.1f}%)\n"
                    )
                    f.write(
                        f"  Fail: {conv_stats['fail']} ({conv_stats['fail_rate']:.1f}%)\n"
                    )
                    f.write(
                        f"  Error: {conv_stats['error']} ({conv_stats['error_rate']:.1f}%)\n"
                    )
                    f.write("\n")
                    
            # Configuration Parameters
            if self.system_config is not None:
                f.write("Configuration Parameters:\n")
                f.write("-" * 25 + "\n")

                # Get configured sections to include
                included_sections = self._get_included_config_sections()

                # Iterate through specified configuration sections
                for field_name in self.system_config.model_fields.keys():
                    # Skip sections not in the included list
                    if field_name not in included_sections:
                        continue

                    field_value = getattr(self.system_config, field_name)

                    # Format section name nicely
                    section_name = field_name.replace("_", " ").title()
                    f.write(f"\n{section_name}:\n")

                    # Use dynamic formatter for the section
                    lines = self._format_config_section(field_value, indent=2)
                    for line in lines:
                        f.write(f"{line}\n")

                f.write("\n")

        return txt_file
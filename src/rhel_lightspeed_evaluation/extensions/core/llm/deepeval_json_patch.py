"""Backport reasoning-tag JSON parsing to deployment-pinned DeepEval 3.9.1.

Adapted from chenrocky/deepeval commit 6c05a26f7be04dc910569690bf8f8690e7a7715b.
Keep upstream error types and metric.error handling by delegating failed parses.
"""

import json
import re
import sys
from collections.abc import Callable
from functools import wraps
from typing import Any

_applied = False


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}") + 1
    if end == 0 and start != -1:
        text += "}"
        end = len(text)
    return text[start:end] if start != -1 and end != 0 else ""


def _parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Valid string values containing ', }' or ', ]' must remain untouched.
        return json.loads(re.sub(r",\s*([\]}])", r"\1", text))


def _strip_think_tags(text: str) -> str:
    # Skip quoted JSON strings even when a preceding reasoning block broke parsing.
    quoted_string = r'"(?:\\.|[^"\\])*"'
    cleaned = re.sub(
        quoted_string + r"|<think>.*?</think>",
        lambda match: match[0] if match[0].startswith('"') else "",
        text,
        flags=re.DOTALL,
    )
    if cleaned == text:
        for match in re.finditer(quoted_string + r"|</think>", text, flags=re.DOTALL):
            if match[0] == "</think>":
                return text[match.end() :]
    return cleaned


def _wrap_parser(parser: Callable[..., Any], *, dataset: bool) -> Callable[..., Any]:
    @wraps(parser)
    def parse(input_string: str, *args: Any, **kwargs: Any) -> Any:
        candidate = input_string.strip() if dataset else _extract_json(input_string)
        try:
            return _parse_json(candidate)
        except json.JSONDecodeError:
            pass

        # Only strip after parsing fails, preserving literal tags in valid JSON.
        cleaned = _strip_think_tags(input_string)
        if cleaned != input_string:
            if dataset:
                try:
                    return _parse_json(cleaned.strip())
                except json.JSONDecodeError:
                    pass
            try:
                return _parse_json(_extract_json(cleaned))
            except json.JSONDecodeError:
                pass
        return parser(input_string, *args, **kwargs)

    return parse


def apply_deepeval_json_patch() -> None:
    """Patch parser definitions and existing DeepEval imports once per process."""
    global _applied
    if _applied:
        return

    from deepeval.dataset import utils as dataset_utils
    from deepeval.metrics import utils as metric_utils
    from deepeval.models.llms import utils as model_utils

    # shortcut: remove when the deployment pin includes the upstream parsing fix.
    for owner, name, dataset in (
        (dataset_utils, "trimAndLoadJson", True),
        (metric_utils, "trimAndLoadJson", False),
        (model_utils, "trim_and_load_json", False),
    ):
        original = getattr(owner, name)
        patched = _wrap_parser(original, dataset=dataset)
        # Providers and GEval import these functions by name before initialization.
        for module_name, module in list(sys.modules.items()):
            if module_name.startswith("deepeval.") and getattr(module, name, None) is original:
                setattr(module, name, patched)
    _applied = True

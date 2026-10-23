"""Reasoning-tag parsing through DeepEval's real provider and metric entry points."""

import asyncio
import json
import multiprocessing
import subprocess
import sys
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import BaseModel

from rhel_lightspeed_evaluation.extensions.core.llm.patches import apply_patches

RAW = '<think>\nCompare {"score": 1} with {"score": 2}.\n</think>\n{"score": 9}'


class Score(BaseModel):
    score: int


@pytest.fixture(params=["dataset", "metrics", "models"])
def parser(request):
    apply_patches()
    from deepeval.dataset import utils as dataset
    from deepeval.metrics import utils as metrics
    from deepeval.models.llms import utils as models

    return {
        "dataset": dataset.trimAndLoadJson,
        "metrics": metrics.trimAndLoadJson,
        "models": models.trim_and_load_json,
    }[request.param]


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('{"score": 9}', {"score": 9}),
        (RAW, {"score": 9}),
        ('Comparing {"score": 1}\n</think>\n{"score": 9}', {"score": 9}),
        ('<think>simple reasoning</think>\n{"score": 9}', {"score": 9}),
        ('<think>{"a": {"b": 1}}</think>\n```json\n{"score": 9}\n```', {"score": 9}),
        ('<think>{"a": 1}</think><think>{"b": 2}</think>{"score": 9,}', {"score": 9}),
        (
            '{"note": "literal <think>text</think> and , } and , ]"}',
            {"note": "literal <think>text</think> and , } and , ]"},
        ),
        (
            '<think>{"a": 1}</think>{"note": "literal <think>text</think>"}',
            {"note": "literal <think>text</think>"},
        ),
    ],
)
def test_all_parsers_handle_reasoning_and_preserve_json(parser, raw, expected):
    assert parser(raw) == expected


@pytest.mark.parametrize(
    "raw", ["", "not JSON {[", '<think>{"a": 1}</think>{"score": nope}', "<think>unfinished"]
)
def test_invalid_json_keeps_upstream_exception_type(parser, raw):
    from deepeval.errors import DeepEvalError

    expected_error = DeepEvalError if parser.__name__ == "trim_and_load_json" else ValueError
    with pytest.raises(expected_error):
        parser(raw)


def test_metric_error_state_only_changes_on_failure():
    apply_patches()
    from deepeval.metrics.utils import trimAndLoadJson

    metric = Mock(error=None)
    assert trimAndLoadJson(RAW, metric=metric) == {"score": 9}
    assert metric.error is None
    with pytest.raises(ValueError, match="invalid JSON"):
        trimAndLoadJson("not json", metric=metric)
    assert "invalid JSON" in metric.error


def test_dataset_arrays_and_existing_model_cleanup_are_preserved():
    apply_patches()
    from deepeval.dataset.utils import trimAndLoadJson
    from deepeval.models.llms.utils import trim_and_load_json

    raw = '[{"note": "<think>literal</think> , ]"}]'
    assert trimAndLoadJson(raw) == json.loads(raw)
    assert trimAndLoadJson('<think>{"example": 1}</think>' + raw) == json.loads(raw)
    assert trim_and_load_json('```json\n{"score": 9,}\n```') == {"score": 9}
    assert trim_and_load_json('{"score": 9') == {"score": 9}


@pytest.mark.parametrize("asynchronous", [False, True])
def test_litellm_model_uses_patched_parser(monkeypatch, asynchronous):
    apply_patches()
    import litellm
    from deepeval.models import LiteLLMModel

    response = litellm.ModelResponse(choices=[{"message": {"content": RAW, "role": "assistant"}}])
    completion = Mock(return_value=response)
    acompletion = AsyncMock(return_value=response)
    monkeypatch.setattr(litellm, "completion", completion)
    monkeypatch.setattr(litellm, "acompletion", acompletion)
    model = LiteLLMModel(model="openai/gpt-4o-mini", api_key="test-key")
    monkeypatch.setattr(model, "calculate_cost", lambda _: 0.25)
    if asynchronous:
        result, cost = asyncio.run(model.a_generate("test prompt", schema=Score))
        acompletion.assert_awaited_once()
        completion.assert_not_called()
    else:
        result, cost = model.generate("test prompt", schema=Score)
        completion.assert_called_once()
        acompletion.assert_not_called()
    assert result == Score(score=9)
    assert cost == 0.25


@pytest.mark.parametrize("asynchronous", [False, True])
def test_metric_schema_fallback_uses_patched_parser(asynchronous):
    apply_patches()
    from deepeval.metrics import utils

    metric = Mock(using_native_model=False, error=None)
    metric.model.generate_with_schema.return_value = RAW
    metric.model.a_generate_with_schema = AsyncMock(return_value=RAW)
    kwargs = {
        "extract_schema": lambda result: result.score,
        "extract_json": lambda result: result["score"],
    }
    if asynchronous:
        result = asyncio.run(
            utils.a_generate_with_schema_and_extract(metric, "prompt", Score, **kwargs)
        )
    else:
        result = utils.generate_with_schema_and_extract(metric, "prompt", Score, **kwargs)
    assert result == 9
    assert metric.error is None


def test_existing_aliases_and_idempotent_initialization():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
from deepeval.dataset import dataset, utils as dataset_utils
from deepeval.metrics.g_eval import g_eval
from deepeval.metrics import utils as metric_utils
from deepeval.models.llms import litellm_model, utils as model_utils
from rhel_lightspeed_evaluation.extensions.core.llm.patches import apply_patches
old = model_utils.trim_and_load_json
apply_patches()
patched = model_utils.trim_and_load_json
assert patched is not old
apply_patches()
assert model_utils.trim_and_load_json is patched
assert litellm_model.trim_and_load_json is patched
assert g_eval.trimAndLoadJson is metric_utils.trimAndLoadJson
assert dataset.trimAndLoadJson is dataset_utils.trimAndLoadJson
assert patched('<think>{"a": 1}</think>{"score": 9}') == {"score": 9}
""",
        ],
        check=True,
        timeout=60,
    )


def _spawned_parse(connection):
    apply_patches()
    from deepeval.models.llms.litellm_model import trim_and_load_json

    connection.send(trim_and_load_json(RAW))
    connection.close()


def test_spawned_worker_initializes_parser_patch():
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    worker = context.Process(target=_spawned_parse, args=(sender,))
    worker.start()
    sender.close()
    try:
        assert receiver.poll(30), "spawned worker did not return a parsed result"
        assert receiver.recv() == {"score": 9}
        worker.join(10)
        assert worker.exitcode == 0
    finally:
        if worker.is_alive():
            worker.kill()
            worker.join(10)
        receiver.close()

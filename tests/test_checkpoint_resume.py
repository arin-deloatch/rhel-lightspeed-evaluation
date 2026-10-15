"""Exercise durable resume using real reports and an abruptly killed worker."""

import argparse
import csv
import json
import multiprocessing
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from lightspeed_evaluation.core.models import EvaluationData, EvaluationResult
from lightspeed_evaluation.core.storage import FileBackendConfig

from rhel_lightspeed_evaluation.extensions.core.models.system import SystemConfigExt
from rhel_lightspeed_evaluation.extensions.pipeline.evaluation.checkpoint import CheckpointStore
from rhel_lightspeed_evaluation.extensions.pipeline.evaluation.pipeline import EvaluationPipelineExt
from rhel_lightspeed_evaluation.extensions.runner.evaluation import run_evaluation


@pytest.fixture
def args(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    config = {
        "core": {"cache_enabled": False, "max_threads": 1},
        "api": {"enabled": False},
        "visualization": {"enabled_graphs": []},
        "storage": [{"type": "file", "enabled_outputs": ["csv", "json"]}],
    }
    (tmp_path / "system.yaml").write_text(yaml.safe_dump(config))
    data = [
        {
            "conversation_group_id": identifier,
            "tag": ["b", "a"],
            "turns": [
                {
                    "turn_id": "t1",
                    "query": "question",
                    "response": "answer",
                    "turn_metrics": ["custom:tool_eval"],
                    "tool_calls": [[{"tool_name": "search", "arguments": {}}]],
                    "expected_tool_calls": [[{"tool_name": "search", "arguments": {}}]],
                }
            ],
        }
        for identifier in ("first", "second")
    ]
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(data))
    return argparse.Namespace(
        system_config=str(tmp_path / "system.yaml"),
        eval_data=str(tmp_path / "data.yaml"),
        output_dir=str(tmp_path / "output"),
        resume=True,
        cache_warmup=False,
        tags=None,
        conv_ids=None,
        metrics=None,
    )


def _killed_worker(args, committed, blocked):
    """Block after the first commit so the parent can SIGKILL a real runner."""
    original_save = CheckpointStore.save
    original_process = EvaluationPipelineExt._process_conversation

    def process(self, conversation):
        if conversation.conversation_group_id == "second":
            blocked.wait(120)
        conversation.turns[0].response = "generated answer"
        conversation.turns[0].contexts = ["retrieved document"]
        conversation.turns[0].api_input_tokens = 17
        return original_process(self, conversation)

    def save(self, conversation, results):
        original_save(self, conversation, results)
        committed.set()
        blocked.wait(120)

    EvaluationPipelineExt._process_conversation = process
    CheckpointStore.save = save
    run_evaluation(args)


def test_sigkill_resume_restores_results_and_amended_data(args, monkeypatch):
    context = multiprocessing.get_context("spawn")
    committed, blocked = context.Event(), context.Event()
    worker = context.Process(target=_killed_worker, args=(args, committed, blocked))
    worker.start()
    try:
        assert committed.wait(60), "child did not persist its first conversation"
    finally:
        worker.kill()
        worker.join(10)
    assert not worker.is_alive()
    checkpoint_dir = Path(args.output_dir) / ".rhel-checkpoint/conversations"
    assert len(list(checkpoint_dir.glob("conversation-*.json"))) == 1
    called = []
    captured = []
    original = EvaluationPipelineExt._process_conversation
    original_process_data = EvaluationPipelineExt._process_eval_data

    def process(self, conversation):
        called.append(conversation.conversation_group_id)
        return original(self, conversation)

    def process_data(self, data):
        results = original_process_data(self, data)
        captured.extend(data)
        return results

    monkeypatch.setattr(EvaluationPipelineExt, "_process_conversation", process)
    monkeypatch.setattr(EvaluationPipelineExt, "_process_eval_data", process_data)
    expected = {"TOTAL": 2, "PASS": 2, "FAIL": 0, "ERROR": 0, "SKIPPED": 0}
    assert run_evaluation(args) == expected
    assert called == ["second"]
    assert captured[0].turns[0].response == "generated answer"
    assert captured[0].turns[0].contexts == ["retrieved document"]
    assert captured[0].turns[0].api_input_tokens == 17
    called.clear()
    assert run_evaluation(args) == expected
    assert called == []
    for report in Path(args.output_dir).glob("evaluation_*_detailed.csv"):
        with report.open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 2
        assert {row["conversation_group_id"] for row in rows} == {"first", "second"}


def test_failed_conversation_does_not_lose_other_completions(args, monkeypatch):
    original = EvaluationPipelineExt._process_conversation

    def process(self, conversation):
        if conversation.conversation_group_id == "first":
            raise RuntimeError("interrupted conversation")
        return original(self, conversation)

    monkeypatch.setattr(EvaluationPipelineExt, "_process_conversation", process)
    assert run_evaluation(args) is None
    files = list(Path(args.output_dir).rglob("conversation-*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text())["conversation"]["conversation_group_id"] == "second"
    called = []

    def retry(self, conversation):
        called.append(conversation.conversation_group_id)
        return original(self, conversation)

    monkeypatch.setattr(EvaluationPipelineExt, "_process_conversation", retry)
    assert run_evaluation(args)["TOTAL"] == 2
    assert called == ["first"]


@pytest.mark.parametrize(
    "change", ["query", "model", "metrics", "duplicate", "corrupt", "no_resume"]
)
def test_resume_rejects_incompatible_or_corrupt_state(args, monkeypatch, change):
    assert run_evaluation(args)["TOTAL"] == 2
    if change in ("query", "duplicate"):
        path = Path(args.eval_data)
        data = yaml.safe_load(path.read_text())
        if change == "query":
            data[0]["turns"][0]["query"] = "changed question"
        else:
            data[1]["conversation_group_id"] = "first"
        path.write_text(yaml.safe_dump(data))
    elif change == "model":
        path = Path(args.system_config)
        config = yaml.safe_load(path.read_text())
        config["llm"] = {"model": "different-judge"}
        path.write_text(yaml.safe_dump(config))
    elif change == "metrics":
        args.metrics = ["custom:keyword_eval"]
    elif change == "corrupt":
        next(Path(args.output_dir).rglob("conversation-*.json")).write_text('{"truncated":')
    else:
        args.resume = False
    monkeypatch.setattr(
        EvaluationPipelineExt,
        "_process_conversation",
        lambda *_: pytest.fail("must reject before provider work"),
    )
    assert run_evaluation(args) is None


def test_atomic_commit_failure_can_be_retried(tmp_path, monkeypatch):
    conversation = EvaluationData(
        conversation_group_id="../unsafe/path", turns=[{"turn_id": "t", "query": "q"}]
    )
    result = EvaluationResult(
        conversation_group_id=conversation.conversation_group_id,
        metric_identifier="custom:tool_eval",
        result="ERROR",
        reason="judge error",
        judge_scores=[
            {
                "judge_id": "judge-a",
                "score": None,
                "reason": "unavailable",
                "judge_input_tokens": 23,
            }
        ],
    )
    with CheckpointStore(tmp_path / "checkpoint") as store:
        store.validate(SystemConfigExt(), [conversation])
        with monkeypatch.context() as patch:
            patch.setattr("os.replace", Mock(side_effect=OSError("disk full")))
            with pytest.raises(OSError, match="disk full"):
                store.save(conversation, [result])
        assert store.load([conversation]) == {}
        store.save(conversation, [result])
        assert store.load([conversation])[conversation.conversation_group_id].results == [result]
        with (
            pytest.raises(ValueError, match="Another evaluation"),
            CheckpointStore(store.directory),
        ):
            pytest.fail("concurrent writer acquired lock")
    with CheckpointStore(tmp_path / "checkpoint") as reopened:
        assert len(reopened.load([conversation])) == 1


def test_checkpoint_io_failure_fails_run_and_preserves_prior_work(args, monkeypatch):
    save = CheckpointStore.save

    def fail_second(self, conversation, results):
        if conversation.conversation_group_id == "second":
            raise OSError("disk full")
        save(self, conversation, results)

    monkeypatch.setattr(CheckpointStore, "save", fail_second)
    assert run_evaluation(args) is None
    assert len(list(Path(args.output_dir).rglob("conversation-*.json"))) == 1
    monkeypatch.setattr(CheckpointStore, "save", save)
    assert run_evaluation(args)["TOTAL"] == 2


def test_script_and_runtime_changes_invalidate_checkpoint(tmp_path, monkeypatch):
    script = tmp_path / "verify.sh"
    script.write_text("exit 0\n")
    conversation = EvaluationData(
        conversation_group_id="test",
        setup_script=script,
        turns=[{"turn_id": "t", "query": "q"}],
    )
    with CheckpointStore(tmp_path / "checkpoint") as store:
        store.validate(SystemConfigExt(), [conversation])
        script.write_text("exit 1\n")
        with pytest.raises(ValueError, match="does not match"):
            store.validate(SystemConfigExt(), [conversation])
        script.write_text("exit 0\n")
        from rhel_lightspeed_evaluation.extensions.pipeline.evaluation import checkpoint

        monkeypatch.setattr(
            checkpoint, "distribution", lambda _: Mock(version="changed", read_text=lambda _: None)
        )
        with pytest.raises(ValueError, match="does not match"):
            store.validate(SystemConfigExt(), [conversation])


@pytest.mark.parametrize("option", ["output", "warmup", "legacy", "storage"])
def test_resume_rejects_unsupported_options(args, option):
    if option == "output":
        args.output_dir = None
    elif option == "warmup":
        args.cache_warmup = True
    elif option == "legacy":
        Path(args.output_dir).mkdir()
        (Path(args.output_dir) / ".completed_ids").write_text("first\n")
    else:
        from rhel_lightspeed_evaluation.extensions.pipeline.evaluation.checkpoint import (
            validate_storage,
        )

        config = SystemConfigExt()
        config.storage = [FileBackendConfig(), Mock()]
        with pytest.raises(ValueError, match="file-only"):
            validate_storage(config)
        return
    assert run_evaluation(args) is None

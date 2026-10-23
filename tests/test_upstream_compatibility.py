"""Regression checks against the pinned upstream, without provider calls."""

import argparse
import json

import httpx
import pytest
import yaml

from rhel_lightspeed_evaluation.extensions.core.api import APIClientExt
from rhel_lightspeed_evaluation.extensions.core.models.system import APIConfigExt, SystemConfigExt
from rhel_lightspeed_evaluation.extensions.runner.evaluation import run_evaluation


def test_runner_defers_pipeline_import_until_judge_patches_are_installed():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import rhel_lightspeed_evaluation.extensions.runner.evaluation
assert 'lightspeed_evaluation.core.metrics.deepeval' not in sys.modules
from rhel_lightspeed_evaluation.extensions.core.llm.patches import apply_patches
apply_patches()
from lightspeed_evaluation.core.metrics.deepeval import DeepEvalLLMManager
from rhel_lightspeed_evaluation.extensions.core.llm.deepeval_patch import DeepEvalLLMManager as Patched
assert DeepEvalLLMManager is Patched
""",
        ],
        check=True,
        timeout=60,
    )


@pytest.mark.parametrize("endpoint", ["query", "responses", "chat/completions"])
def test_request_params_and_endpoint_routing(endpoint):
    config = APIConfigExt(
        api_base="http://stack/api/lightspeed",
        endpoint_type=endpoint,
        cache_enabled=False,
        extra_request_params={"no_tools": True},
    )
    client = APIClientExt(config)
    request = client._prepare_request("question", extra_request_params={"no_tools": False})
    payload = client._serialize_request(request)
    assert payload["no_tools"] is False
    assert ("messages" in payload) == (endpoint == "chat/completions")
    client.close()


def test_infer_path_and_native_rag_metadata():
    client = APIClientExt(
        APIConfigExt(
            api_base="http://stack/api/lightspeed",
            endpoint_type="infer",
            cache_enabled=False,
        )
    )
    client.client.close()
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": {
                    "text": "answer",
                    "request_id": "request-1",
                    "input_tokens": 7,
                    "output_tokens": 4,
                    "tool_calls": [{"id": "call-1", "name": "search", "args": {"q": "RHEL"}}],
                    "tool_results": [{"id": "call-1", "type": "mcp_call", "content": "fallback"}],
                    "rag_chunks": [{"content": "native context", "source": "docs"}],
                }
            },
        )

    client.client = httpx.Client(
        base_url=client.config.api_base, transport=httpx.MockTransport(respond)
    )
    try:
        response = client.query("question")
    finally:
        client.close()
    assert requests[0].url.path == "/api/lightspeed/v1/infer"
    assert json.loads(requests[0].content) == {"question": "question", "include_metadata": True}
    assert response.contexts == ["native context"]
    assert response.tool_calls[0][0]["result"] == "fallback"
    assert (response.input_tokens, response.output_tokens) == (7, 4)


def test_config_roundtrip_preserves_all_agent_types():
    config = SystemConfigExt.model_validate(
        {
            "agents": {
                "default": {"agent": ["chat", "cluster"]},
                "chat": {"type": "http_api", "endpoint_type": "chat/completions"},
                "cluster": {"type": "proposal", "namespace": "test"},
            }
        }
    )
    restored = SystemConfigExt.model_validate(config.model_dump())
    assert restored.agents.agents["chat"].endpoint_type == "chat/completions"
    assert restored.agents.agents["cluster"].namespace == "test"


def _args(tmp_path, config):
    config_path = tmp_path / "system.yaml"
    config_path.write_text(yaml.safe_dump(config))
    data_path = tmp_path / "data.yaml"
    data_path.write_text(
        yaml.safe_dump(
            [
                {
                    "conversation_group_id": "test",
                    "conversation_metrics": [],
                    "turns": [
                        {
                            "turn_id": "t1",
                            "query": "question",
                            "response": "answer",
                            "tool_calls": [[{"tool_name": "search", "arguments": {}}]],
                            "expected_tool_calls": [[{"tool_name": "search", "arguments": {}}]],
                            "turn_metrics": ["custom:tool_eval"],
                        }
                    ],
                }
            ]
        )
    )
    return argparse.Namespace(
        system_config=str(config_path),
        eval_data=str(data_path),
        output_dir=str(tmp_path / "output"),
        cache_warmup=False,
        tags=None,
        conv_ids=None,
        metrics=None,
    )


def test_offline_runner_and_reports(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")
    args = _args(
        tmp_path,
        {
            "core": {"cache_enabled": False},
            "api": {"enabled": False},
            "llm": {"provider": "openai", "model": "gpt-4o-mini"},
            "visualization": {"enabled_graphs": []},
        },
    )
    assert run_evaluation(args) == {"TOTAL": 1, "PASS": 1, "FAIL": 0, "ERROR": 0, "SKIPPED": 0}
    reports = list((tmp_path / "output").glob("evaluation_*_summary.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["summary_stats"]["overall"]["PASS"] == 1


@pytest.mark.parametrize(
    "parallel, repeat, agent_count", [(False, 1, 1), (False, 2, 1), (True, 2, 1), (True, 2, 2)]
)
@pytest.mark.parametrize("resume", [False, True])
def test_agent_repeats_use_rhel_driver_and_write_reports(
    tmp_path, monkeypatch, parallel, repeat, agent_count, resume
):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(
                (self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            )
            body = json.dumps(
                {
                    "data": {
                        "text": "answer",
                        "request_id": "request-1",
                        "tool_calls": [{"name": "search", "args": {}}],
                        "tool_results": [],
                    }
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    args = _args(
        tmp_path,
        {
            "core": {"cache_enabled": False},
            "llm": {"provider": "openai", "model": "gpt-4o-mini"},
            "visualization": {"enabled_graphs": []},
            "agents": {
                "enabled": True,
                "default": {"agent": ["stack"], "repeat": repeat, "parallel": parallel},
                "stack": {
                    "type": "http_api",
                    "endpoint_type": "infer",
                    "api_base": f"http://127.0.0.1:{server.server_port}/api/lightspeed",
                },
            },
        },
    )
    args.resume = resume
    if agent_count == 2:
        from pathlib import Path

        path = Path(args.system_config)
        config = yaml.safe_load(path.read_text())
        config["agents"]["other"] = dict(config["agents"]["stack"])
        config["agents"]["default"]["agent"].append("other")
        path.write_text(yaml.safe_dump(config))
    expected_runs = repeat * agent_count
    try:
        summary = run_evaluation(args)
        assert summary is not None
        assert {key: summary[key] for key in ("TOTAL", "PASS", "FAIL", "ERROR", "SKIPPED")} == {
            "TOTAL": expected_runs,
            "PASS": expected_runs,
            "FAIL": 0,
            "ERROR": 0,
            "SKIPPED": 0,
        }
        if resume:
            assert len(requests) == expected_runs
            repeated = run_evaluation(args)
            assert repeated == summary
            assert len(requests) == expected_runs
            assert len(list((tmp_path / "output").rglob("conversation-*.json"))) == expected_runs
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert len(requests) == expected_runs
    assert all(path == "/api/lightspeed/v1/infer" for path, _ in requests)
    run_directories = list((tmp_path / "output").glob("eval_*/*/run_*"))
    assert len(run_directories) == expected_runs
    for run in run_directories:
        reports = list(run.glob("evaluation_*_detailed.csv"))
        assert reports
        for report in reports:
            import csv

            with report.open() as stream:
                assert len(list(csv.DictReader(stream))) == 1

    if repeat == 1 and not resume:
        assert len(list((tmp_path / "output").glob("evaluation_*_detailed.csv"))) == 1


def test_failed_agent_run_is_not_reported_as_success(tmp_path, monkeypatch):
    from lightspeed_evaluation.pipeline.behavioral.models import RunResult

    from rhel_lightspeed_evaluation.extensions.runner import orchestrator

    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")
    monkeypatch.setattr(
        orchestrator,
        "run",
        lambda *_args, **_kwargs: [
            RunResult(
                agent_name="stack",
                run_index=1,
                output_dir="",
                success=False,
                error="connection failed",
            )
        ],
    )
    args = _args(tmp_path, {"api": {"enabled": True}, "core": {"cache_enabled": False}})
    assert run_evaluation(args) is None


@pytest.mark.parametrize("conversation_level", [False, True])
def test_metric_exception_preserves_diagnostics_and_continues(conversation_level):
    from unittest.mock import Mock

    from lightspeed_evaluation.core.models import EvaluationData, EvaluationResult
    from lightspeed_evaluation.pipeline.evaluation.errors import EvaluationErrorHandler

    from rhel_lightspeed_evaluation.extensions.pipeline.evaluation.processor import (
        ConversationProcessorExt,
    )

    data = EvaluationData.model_validate(
        {
            "conversation_group_id": "test",
            "tag": ["regression"],
            "turns": [{"turn_id": "t1", "query": "question", "response": "answer"}],
        }
    )
    components = Mock(error_handler=EvaluationErrorHandler())
    success = EvaluationResult(
        conversation_group_id="test", metric_identifier="custom:tool_eval", result="PASS"
    )
    components.metrics_evaluator.evaluate_metric.side_effect = [
        RuntimeError("judge failed"),
        success,
    ]
    processor = ConversationProcessorExt(Mock(), components)
    metrics = ["ragas:faithfulness", "custom:tool_eval"]
    if conversation_level:
        results = processor._evaluate_conversation(data, metrics)
    else:
        results = processor._evaluate_turn(data, 0, data.turns[0], metrics)
    assert len(results) == 2
    failure, passed = results
    assert failure.result == "ERROR"
    assert failure.reason == "RuntimeError: judge failed"
    assert failure.metric_identifier == "ragas:faithfulness"
    assert failure.conversation_group_id == "test"
    assert failure.tag == {"regression"}
    assert failure.turn_id == (None if conversation_level else "t1")
    assert failure.query == ("" if conversation_level else "question")
    assert passed is success


def test_runner_saves_error_and_continues_after_metric_exception(tmp_path, monkeypatch):
    from lightspeed_evaluation.pipeline.evaluation.evaluator import MetricsEvaluator

    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")
    original = MetricsEvaluator.evaluate_metric

    def evaluate(self, request):
        if request.turn_idx == 0:
            raise RuntimeError("judge failed")
        return original(self, request)

    monkeypatch.setattr(MetricsEvaluator, "evaluate_metric", evaluate)
    args = _args(tmp_path, {"api": {"enabled": False}, "core": {"cache_enabled": False}})
    from pathlib import Path

    data_path = Path(args.eval_data)
    data = yaml.safe_load(data_path.read_text())
    data[0]["turns"].append({**data[0]["turns"][0], "turn_id": "t2"})
    data_path.write_text(yaml.safe_dump(data))
    assert run_evaluation(args) == {"TOTAL": 2, "PASS": 1, "FAIL": 0, "ERROR": 1, "SKIPPED": 0}
    report = json.loads(next((tmp_path / "output").glob("evaluation_*_summary.json")).read_text())
    assert report["summary_stats"]["overall"]["ERROR"] == 1


def test_watsonx_patch_retains_upstream_retry_setup(monkeypatch):
    import lightspeed_evaluation.core.llm.deepeval as upstream

    from rhel_lightspeed_evaluation.extensions.core.llm.deepeval_patch import DeepEvalLLMManager

    calls = {}
    monkeypatch.setenv("WATSONX_API_KEY", "test-watsonx-key")
    monkeypatch.delenv("WX_API_KEY", raising=False)
    monkeypatch.setattr(
        DeepEvalLLMManager,
        "_patch_deepeval_retries",
        lambda _self, retries: calls.update(retries=retries),
    )
    monkeypatch.setattr(upstream, "LiteLLMModel", lambda **kwargs: calls.update(kwargs))
    original = {"num_retries": 2, "timeout": 30, "parameters": {"temperature": 0}}
    DeepEvalLLMManager("watsonx/test-model", original)
    assert calls["retries"] == 2
    assert calls["api_key"] == "test-watsonx-key"
    assert calls["temperature"] == 0
    assert original["parameters"] == {"temperature": 0}

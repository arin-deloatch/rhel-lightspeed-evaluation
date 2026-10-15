<h1 align="center">RHEL Lightspeed Evaluation</h1>

<p align="center"><i>An evaluation project for RHEL Lightspeed using the <a href="https://github.com/lightspeed-core/lightspeed-evaluation" target="_blank"> LSC Evaluation Framework</a>.</i></p>

---

**This repository is under heavy construction. We're actively adding evaluation scenarios, refining metrics, and expanding test coverage.**

## Overview

This project provides evaluation configurations and test data specifically for RHEL Lightspeed, leveraging the comprehensive evaluation capabilities of the lightspeed-evaluation framework including:

- **Multi-Framework Metrics**: Ragas, DeepEval, and custom evaluations
- **Custom Runtime Metrics**: Define GEval metrics inline or in system config
- **Panel of Judges**: Multi-model evaluation for robust assessment
- **Turn & Conversation-Level Analysis**: Individual queries and multi-turn conversations
- **Multiple Evaluation Types**: Response quality, context relevance, tool calls, and script-based verification
- **Flexible LLM Support**: OpenAI, Watsonx, Gemini, Vertex AI, vLLM and others
- **Rich Reporting**: CSV, JSON, TXT reports with visualization graphs

## Quick Start

### Installation

```bash
# Install dependencies including dev tools
make install-dev
```

#### Upstream 0.8 compatibility

The tested upstream dependency is declared in [pyproject.toml](pyproject.toml)
as a direct Git requirement, so the exact commit applies to downstream
installations as well as this repository's `uv.lock`. Normal installs use the
recorded dependencies; no deployment repository is required.

To synchronize the environment and run compatibility tests using the recorded
pin, or check manifest/lockfile consistency without changes:

```bash
make sync-evaluation
make check-evaluation-pin
```

To select a different upstream commit, provide its full 40-character lowercase
SHA through an environment variable or make argument:

```bash
export EVALUATION_REV="<full-commit-SHA>"
make sync-evaluation
make check-evaluation-pin
unset EVALUATION_REV
```

Alternatively, use `make sync-evaluation EVALUATION_REV=<full-commit-SHA>`.
Branches, tags, and abbreviated SHAs are rejected. The selected commit becomes
the recorded default in `pyproject.toml`; there is no separate default in the
Makefile or script. `EVALUATION_REV` controls these make commands, not normal
installs or running evaluations. Direct script invocations accept `--revision`.

Sync uses `uv` to update the requirement and lockfile, installs the locked dev
environment, and runs the offline compatibility and checkpoint tests. Check only
reads the manifest and lockfile; it fails on drift from the selected or recorded
commit. CI also checks their consistency. A failed sync leaves changes available
for inspection; it does not establish compatibility with that revision. Review
and commit `pyproject.toml` and `uv.lock` together after tests pass, then run a
smoke test against your deployment.

The RHEL CLI retains upstream argument names, offline evaluation, legacy `api:`
migration, and `agents:` selection/repeats, including process-based parallel runs.
Its HTTP driver supplies `APIClientExt` for default and per-conversation agents;
the upstream OpenShift `proposal` agent type remains available. Supported HTTP endpoints include
`query`, `streaming`, `infer`, `responses`, and the RHEL `chat/completions` extension.
The infer base URL includes `/api/lightspeed` exactly once. Native RAG chunks take
precedence over the fallback extracted from MCP tool results.

The orchestrator reuses upstream planning, filtering, and summary helpers.
It supplies a RHEL worker because upstream does not expose config/pipeline factory
arguments. These private integration points are covered by offline and local HTTP
regression tests and must be checked when updating the pinned upstream revision.
Each worker installs the WatsonX/Vertex judge patches before initializing metrics.
The WatsonX patch delegates retry configuration to the upstream implementation.

The extension also backports the DeepEval reasoning-tag JSON fix from
[chenrocky/deepeval commit 6c05a26](https://github.com/chenrocky/deepeval/commit/6c05a26f7be04dc910569690bf8f8690e7a7715b)
to the currently locked **DeepEval 3.9.1**. Normal RHEL runner initialization
installs it in each process, including spawned workers; no fork overlay or special
install command is needed. It covers the model, metric, and dataset parsers and
their already-imported references, including synchronous/asynchronous LiteLLM and
GEval paths. Valid JSON is parsed before cleanup, preserving literal `<think>`
text and comma sequences in strings. If parsing fails, complete reasoning blocks
or a closing-tag-only reasoning prefix are removed, while quoted JSON content is
preserved. Unparseable responses retain upstream exception types and metric error
reporting. Calls made directly through DeepEval outside the RHEL runner must first
invoke `rhel_lightspeed_evaluation.extensions.core.llm.patches.apply_patches()`.
This changes parsing only, not the judge rubric, model, score, or retry policy.
Remove the backport when the recorded dependencies provide the fix.

The current upstream pin predates upstream PR #304. `ConversationProcessorExt` retains
per-metric exception recovery for turn and conversation metrics, preserving error
diagnostics and continuing the remaining evaluations. Remove that backport when
the upstream pin includes the fix. No upstream package files are patched.

```bash
uv sync --extra dev
uv run pytest tests/ -q
make lint
make type-check
```

Single-agent/single-repeat runs retain flat output copies for existing consumers;
other runs use upstream's nested `eval_<timestamp>/<agent>/run_<n>/` layout. Reports
are written per run; the newer upstream combined multi-run report is not available
at this pin. Worker failures return a failing CLI status instead of a successful empty total. The old
checkpoint/incremental-output fork overlays are not needed: the RHEL runner now
provides opt-in conversation checkpoints, described below.
No provider credentials are needed for the regression tests; they use deterministic
metrics and a local mock HTTP server, including spawned-process tests.

### Basic Configuration

1. **Set up your LLM provider credentials:**
   ```bash
   # For OpenAI
   export OPENAI_API_KEY="your-api-key"

   # For Vertex AI
   export GOOGLE_CLOUD_PROJECT="your-cloud-project"
   export GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
   export GOOGLE_API_KEY=<insert GCP project API individual key>

   # For Watsonx
   export WATSONX_API_KEY="your-api-key"
   export WATSONX_URL="your-instance-url"
   ```

2. **Configure your `config/system.yaml`:**
   ```yaml
   llm:
     provider: vertex  # or openai, watsonx, etc.
     model: gemini-2.0-flash
     temperature: 0.0
     max_tokens: 512
   ```

### Running Evaluations

#### Using the CLI

```bash
# Run evaluation with default paths
rls-evaluate

# Specify custom paths
rls-evaluate \
  --system-config config/system.yaml \
  --eval-data config/evaluation_data.yaml

# Override output directory
rls-evaluate \
  --system-config config/system.yaml \
  --eval-data config/evaluation_data.yaml \
  --output-dir ./my_results
```



### Checkpoint and resume

Use `--resume` **from the first invocation** and give the run a fresh, explicit
output directory. Repeat exactly the same command after an interruption:

```bash
uv run --locked rls-evaluate \
  --system-config config/system.yaml \
  --eval-data config/evaluation_data.yaml \
  --output-dir eval_output/resumable-run \
  --resume
```

Completed conversations are written atomically under `.rhel-checkpoint/`, with
all metric results and the amended conversation (responses, retrieved contexts,
tool calls, token counts). Each write is flushed to disk before it counts as
complete. A restart restores this data, skips completed conversations, evaluates
the remainder, and rebuilds reports with both saved and new results. No upstream
package files are modified. Python callers can pass `resume=True` to
`EvaluationPipelineExt.run_evaluation`, or set `resume=True` on the runner's
argument namespace.

For agents/repeats, checkpoints and reports are separated under
`eval_resume/<agent>/run_<n>/`, including parallel process runs. A single
agent/repeat also gets flat report copies; checkpoints remain in their original
run directory. Offline evaluations keep their checkpoints directly below the
output directory. An OS lock prevents concurrent writers using the same run.

Resume rejects changes to the validated input, effective configuration, dataset
metadata, verification-script contents, and recorded framework versions/Git
provenance. It requires unique conversation IDs and **file-only storage**, on a
local Linux/macOS filesystem supporting file locks and atomic rename. Do not
change the deployed model, external resources, environment-based model settings,
or editable source code between attempts; those changes cannot all be detected
from the saved fingerprint. Use a new output directory for a new experiment.

The recovery boundary is a whole conversation. Work still in flight when the
process dies may run again, including provider calls. Completed `ERROR`/`SKIPPED`
results are preserved, not automatically retried; use a separate recovery run
to reevaluate failures. A worker exception fails the invocation after saving
other completed conversations. A checkpoint/storage failure also fails the run.
This is not a guarantee of exactly-once remote API execution.

Reports retain upstream timestamped names. An interrupted attempt may leave
partial reports: after a successful resume, use the **newest complete report per
agent/repeat**, not a merge of reports from successive attempts. Keep the entire
output directory until the run is finished. Checkpoints contain the same private
evaluation content as the reports and should be handled accordingly.

Runs without `--resume` retain upstream behavior and refuse directories containing
RHEL checkpoints. Legacy fork `.completed_ids`/`.results.jsonl` files cannot be
imported. Cache warmup cannot be combined with resume. The `ls-evaluation-config`
batch wrapper does not yet expose this flag; use `rls-evaluate` with a fixed batch
YAML to exercise this extension independently before integrating that wrapper.

Validation without provider credentials:

```bash
LITELLM_LOCAL_MODEL_COST_MAP=True DEEPEVAL_TELEMETRY_OPT_OUT=YES \
  uv run --locked pytest tests/ -q
make lint type-check
make check-evaluation-pin
```

The tests kill a worker after a durable save, restart it, and verify that only
unfinished conversations execute and reports have no missing or duplicate rows.
They also cover saved API data, agent/repeat isolation, concurrent writers,
incompatible/corrupt checkpoints, and failed writes.

## Metrics

The framework supports multiple types of custom metrics for evaluating LLM responses:

### Available Metric Types

- **Ragas Metrics**: Response relevancy, faithfulness, context recall, context precision, context relevance
- **DeepEval Metrics**: Conversation completeness, conversation relevancy, knowledge retention
- **Custom Metrics**: Answer correctness, intent evaluation, tool call evaluation
- **Script-Based Metrics**: Infrastructure/environment validation
- **GEval Metrics**: Flexible LLM-as-judge metrics with custom criteria

### GEval Metrics

GEval metrics allow you to define custom evaluation criteria that are assessed by an LLM judge. You can define them in two ways:

#### 1. System-Level Metrics (Reusable)

Define metrics in `config/system.yaml` under `metrics_metadata` for reuse across all evaluations:

```yaml
# config/system.yaml
metrics_metadata:
  turn_level:
    "geval:technical_accuracy":
      criteria: |
        Assess whether the response provides technically accurate information,
        commands, code, syntax, and follows relevant best practices.
      evaluation_params:
        - query
        - response
        - expected_response
      evaluation_steps:
        - "Verify that the provided syntax is valid"
        - "Check if appropriate modules/functions are used"
        - "Assess alignment with best practices"
      threshold: 0.7
      description: "Technical accuracy of commands/code"
```

Use in evaluations by referencing the metric name:

```yaml
# config/evaluation_data.yaml
turns:
  - turn_id: "turn_1"
    query: "How do I check firewall status?"
    response: "Use systemctl status firewalld"
    turn_metrics:
      - geval:technical_accuracy
```

#### 2. Inline Custom Metrics

For one-off evaluations, define GEval metrics directly in your evaluation data:

```yaml
turns:
  - turn_id: "turn_1"
    query: "What monitoring tools are available?"
    response: "You can use top, htop, or sar"
    turn_metrics:
      - geval:custom_monitoring_eval
    turn_metrics_metadata:
      geval:custom_monitoring_eval:
        criteria: |
          Evaluate if the response mentions multiple monitoring
          tools and explains their differences.
        evaluation_steps:
          - "Check if 2+ tools are mentioned"
          - "Verify explanations are provided"
        threshold: 0.7
```

### Panel of Judges

Use multiple LLM judges for more robust evaluation:

```yaml
# config/system.yaml
panel:
  enabled: true
  apply_to:
    - geval
  judges:
    - provider: vertex
      model: gemini-2.0-flash
      temperature: 0.0
    - provider: openai
      model: gpt-4o-mini
      temperature: 0.0
```

## Output
*WIP, still working through ways to integrate panel metrics into the evaluation outputs. Currently, CSV and JSON are supported.*

Evaluation results are saved to `./eval_output` (configurable) with:

- **CSV**: Detailed results with scores, reasons, execution time
- **JSON**: Summary statistics (TOTAL, PASS, FAIL, ERROR counts)
- **TXT**: Human-readable summary report
- **Graphs**: Visual analysis (pass rates, score distributions, heatmaps)

## Configuration Reference

### System Configuration

Key sections in `config/system.yaml`:

```yaml
# Core evaluation parameters
core:
  max_threads: 50

# LLM as a judge configuration
llm:
  provider: vertex
  model: gemini-2.0-flash
  temperature: 0.0
  max_tokens: 512
  cache_enabled: true
  cache_dir: ".caches/llm_cache"

# Embedding configuration
embedding:
  provider: "openai"
  model: "text-embedding-3-small"
  cache_enabled: true
  cache_dir: ".caches/embedding_cache"

# Panel of judges
panel:
  enabled: true
  apply_to:
    - geval
  judges:
    - provider: vertex
      model: gemini-2.0-flash
      temperature: 0.0
    - provider: openai
      model: gpt-4o-mini
      temperature: 0.0

# Lightspeed-stack API Configuration
api:
  enabled: false
  api_base: http://localhost:8080
  endpoint_type: streaming
  cache_enabled: true
  cache_dir: ".caches/api_cache"

# Metrics metadata (define reusable metrics)
metrics_metadata:
  turn_level:
    "geval:technical_accuracy":
      criteria: "..."
      evaluation_steps: []
      threshold: 0.7
  conversation_level:
    "geval:conversation_coherence":
      criteria: "..."
      threshold: 0.6

# Output Configuration
output:
  output_dir: "./eval_output"
  enabled_outputs: [csv, json, txt]
  csv_columns:
    - conversation_group_id
    - turn_id
    - metric_identifier
    - score
    - judge_id # Will not populate if "panel" is not enabled
    - threshold
    - result
  # Config sections to include in the summary output file; used for tracking model params/metadata
  summary_config_sections:
    - llm
    - panel_of_judges

# Visualization
visualization:
  enabled_graphs:
    - pass_rates
    - score_distribution
    - conversation_heatmap
    - status_breakdown

# Logging
logging:
  source_level: INFO
  package_level: ERROR
```


## License

This project is licensed under the Apache License 2.0.

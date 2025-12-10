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

### Basic Configuration

1. **Set up your LLM provider credentials:**
   ```bash
   # For OpenAI
   export OPENAI_API_KEY="your-api-key"

   # For Vertex AI
   export GOOGLE_CLOUD_PROJECT="your-cloud-project"
   export GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json

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

<h1 align="center">RHEL Lightspeed Evaluation</h1>
<p align="center"><i>An evaluation project for RHEL Lightspeed using the <a href="https://github.com/lightspeed-core/lightspeed-evaluation" target="_blank"> LSC Evaluation Framework</a>.</i></p>

---

**This repository is under heavy construction. We're actively adding evaluation scenarios, refining metrics, and expanding test coverage.**

## Overview

This project provides evaluation configurations and test data specifically for RHEL Lightspeed, leveraging the comprehensive evaluation capabilities of the lightspeed-evaluation framework including:

- **Multi-Framework Metrics**: Ragas, DeepEval, and custom evaluations
- **GEval Integration**: Runtime-configurable custom evaluation criteria with centralized metric registry
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

1. **Copy the example system configuration:**
   ```bash
   cp config/system/system.example.yaml config/system/system.yaml
   ```

2. **Set up your LLM provider credentials:**
   ```bash
   # For OpenAI
   export OPENAI_API_KEY="your-api-key"

   # For Vertex AI
   export GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json

   # For Watsonx
   export WATSONX_API_KEY="your-api-key"
   export WATSONX_URL="your-instance-url"
   ```

3. **Configure your system.yaml:**
   ```yaml
   llm:
     provider: vertex  # or openai, watsonx, etc.
     model: gemini-2.0-flash
     temperature: 0.0
     max_tokens: 500
   ```

### Running Evaluations

#### Using the CLI

```bash
# Run evaluation with default paths
rls-evaluate

# Specify custom paths
rls-evaluate \
  --system-config config/system/system.yaml \
  --eval-data config/evaluation_data/pseudo_eval_data.yaml

# Override output directory
rls-evaluate \
  --system-config config/system/system.yaml \
  --eval-data config/evaluation_data/pseudo_eval_data.yaml \
  --output-dir ./my_results
```

#### Using Python

```python
from rhel_lightspeed_evaluation.cli import run_evaluation

results = run_evaluation(
    system_config_path="config/system/system.yaml",
    evaluation_data_path="config/scenarios/basic/my_eval.yaml",
    output_dir="./eval_output"
)
```

## GEval Features

### Metric Registry

Define reusable GEval metrics once in `config/registry/geval_metrics.yaml` and reference them across all evaluations:

**Registry Definition:**
```yaml
# config/geval_metrics.yaml
technical_accuracy:
  criteria: |
    Evaluate the technical accuracy of the RHEL command recommendation.
    Consider:
    1. Is the command syntactically correct for RHEL?
    2. Does it accomplish the stated goal?
    3. Does it follow RHEL best practices?
  evaluation_steps:
    - "Verify the command syntax is valid for RHEL"
    - "Confirm the command achieves the intended result"
    - "Assess alignment with RHEL best practices"
  threshold: 0.8

command_safety:
  criteria: |
    Evaluate the safety of the recommended command.
    Consider destructive potential and appropriate warnings.
  threshold: 0.9
```

**Using in Evaluations:**
```yaml
# config/scenarios/my_eval.yaml
conversations:
  - conversation_group_id: "test_001"
    turns:
      - turn_id: "turn_1"
        query: "How do I check firewall status?"
        response: "Use systemctl status firewalld"

        # Simply reference metrics from registry
        turn_metrics:
          - "geval:technical_accuracy"
          - "geval:command_safety"
        # No metadata needed! Loaded from registry
```

### Default Metrics Auto-Apply

Configure default GEval metrics in `system.yaml` to automatically apply them to all evaluations:

```yaml
# config/system.yaml
geval:
  enabled: true
  registry_path: "config/geval_metrics.yaml"

  # Auto-applied to all turns unless overridden
  default_turn_metrics:
    - "geval:technical_accuracy"
    - "geval:command_safety"
    - "geval:security_awareness"

  # Auto-applied to all conversations
  default_conversation_metrics:
    - "geval:conversation_coherence"
    - "geval:conversation_helpfulness"
```

Now you can create evaluations without specifying metrics:

```yaml
conversations:
  - conversation_group_id: "test_001"
    turns:
      - turn_id: "turn_1"
        query: "How do I check firewall status?"
        response: "Use systemctl status firewalld"
        # Metrics auto-applied from system.yaml!
```

### Example Registry Metrics

**Turn-Level Metrics:**
- `geval:technical_accuracy` - Command correctness and accuracy
- `geval:command_safety` - Safety assessment
- `geval:security_awareness` - Security implications
- `geval:completeness` - Response completeness
- `geval:rhel_version_awareness` - RHEL version appropriateness
- `geval:troubleshooting_methodology` - Diagnostic approach quality

**Conversation-Level Metrics:**
- `geval:conversation_coherence` - Flow and consistency
- `geval:conversation_helpfulness` - Problem-solving effectiveness

### Custom Runtime Metrics

For one-off custom metrics, define them inline:

```yaml
turns:
  - turn_id: "turn_1"
    query: "What monitoring tools are available?"
    response: "You can use top, htop, or sar"

    turn_metrics:
      - "geval:technical_accuracy"  # From registry
      - "geval:custom_metric"        # Custom

    turn_metrics_metadata:
      geval:custom_metric:
        criteria: |
          Evaluate if the response mentions multiple monitoring
          tools and explains their differences.
        evaluation_steps:
          - "Check if 2+ tools are mentioned"
          - "Verify explanations are provided"
        threshold: 0.7
```

## Output

Evaluation results are saved to `./eval_output` (configurable) with:

- **CSV**: Detailed results with scores, reasons, execution time
- **JSON**: Summary statistics (TOTAL, PASS, FAIL, ERROR counts)
- **TXT**: Human-readable summary report
- **Graphs**: Visual analysis (pass rates, score distributions, heatmaps)

## Configuration Reference

### System Configuration

Key sections in `config/system.yaml`:

```yaml
# LLM Configuration
llm:
  provider: vertex
  model: gemini-2.0-flash
  temperature: 0.0
  max_tokens: 500
  cache_enabled: true

# GEval Configuration
geval:
  enabled: true
  registry_path: "config/geval_metrics.yaml"
  default_turn_metrics: []
  default_conversation_metrics: []

# Output Configuration
output:
  output_dir: "./eval_output"
  enabled_outputs: [csv, json, txt]

# Logging
logging:
  source_level: INFO
  package_level: ERROR
```


## License

This project is licensed under the Apache License 2.0.

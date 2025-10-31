<h1 align="center">RHEL Lightspeed Evaluation</h1>
<p align="center"><i>An evaluation project for RHEL Lightspeed using the <a href="https://github.com/lightspeed-core/lightspeed-evaluation" target="_blank"> LSC Evaluation Framework</a>.</i></p>

---


**This is repository is under heavy construction. We're actively adding evaluation scenarios, refining metrics, and expanding test coverage.**

## Overview

This project provides evaluation configurations and test data specifically for RHEL Lightspeed, leveraging the comprehensive evaluation capabilities of the lightspeed-evaluation framework including:

- **Multi-Framework Metrics**: Ragas, DeepEval, and custom evaluations
- **Turn & Conversation-Level Analysis**: Individual queries and multi-turn conversations
- **Multiple Evaluation Types**: Response quality, context relevance, tool calls, and script-based verification
- **Flexible LLM Support**: OpenAI, Watsonx, Gemini, vLLM and others
- **Rich Reporting**: CSV, JSON, TXT reports with visualization graphs

## Quick Start

### Installation

```bash
# Install dependencies including dev tools
make install-dev
```

## License

This project is licensed under the Apache License 2.0.

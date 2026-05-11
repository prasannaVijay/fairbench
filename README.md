# FAIRBench

A fairness benchmarking framework for generative AI. FAIRBench systematically evaluates text generation models for bias, representational harm, and stereotype amplification using counterfactual testing and a suite of six quantitative fairness metrics.

```
Scenario Generator → Counterfactual Testing → Model Interface
                                                     ↓
Scorecard Generator ← Metrics Engine ← Output Evaluation
```

---

## Documentation

| Document | Description |
|---|---|
| [Configuring a Benchmark Spec](docs/benchmark-spec.md) | Full field reference for the input YAML — model, scenarios, metrics, output settings |
| [Reading Your Scorecard](docs/reading-your-scorecard.md) | How to interpret the model card output — bands, reasoning sections, and what to do next |
| [Benchmark template](examples/benchmark_template.yaml) | Annotated YAML template to copy and fill in |
| [Example audit](examples/gender_audit.yaml) | A working example targeting gender and service parity |

---

## Installation

Requires Python 3.11+. Install with [uv](https://github.com/astral-sh/uv) (recommended) or pip:

```bash
uv pip install -e ".[dev]"
# or
pip install -e ".[dev]"
```

Set API keys for the models you want to test:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
```

---

## Quick Start

The fastest way to run a fairness audit is to describe your model and scenarios in a single YAML file and pass it to `fairbench run`. Copy the template, fill in your model, and you are done:

```bash
cp examples/benchmark_template.yaml my_audit.yaml
# edit my_audit.yaml — set your model and scenarios
fairbench run my_audit.yaml
```

FAIRBench writes a Markdown model card and a JSON scorecard to `./reports/` by default.

```bash
# Write only the Markdown model card
fairbench run my_audit.yaml --output_format md

# Write only the JSON scorecard
fairbench run my_audit.yaml --output_format json

# Override the output directory
fairbench run my_audit.yaml --output ./my_reports
```

See [Configuring a Benchmark Spec](docs/benchmark-spec.md) for the full field reference, and [Reading Your Scorecard](docs/reading-your-scorecard.md) to understand the output.

---

## CLI Reference

### `fairbench run`

Run a fairness evaluation. Accepts a benchmark spec YAML (recommended), a built-in scenario set name, or a path to a scenario file.

```
fairbench run <scenario_or_spec> [OPTIONS]

Arguments:
  scenario_or_spec  Benchmark spec YAML, built-in scenario name, or scenario file path.
                    A file containing a 'model_under_test' key is treated as a benchmark spec.

Options:
  --model, -m       Model adapter — ignored when a benchmark spec supplies the model.
                    Values: "anthropic", "openai", or a claude-*/gpt-* model string.
  --output, -o      Output directory for scorecard files (default: ./reports).
  --output_format   Scorecard format: json | md | all  (default: all).
  --metrics         Comma-separated list of metrics, e.g. "RSI,HSI,DSI" (default: all six).
  --concurrency, -c Max concurrent API calls (default: 10).
  --verbose, -v     Show full error tracebacks.
```

### `fairbench scorecard`

Generate a scorecard for a previously completed run by its run ID.

```
fairbench scorecard <run_id> [--output scorecard.json]
```

### `fairbench runs`

List recent evaluation runs stored in the local SQLite database.

```
fairbench runs [--limit 20]
```

### `fairbench show`

Show metric results and details for a specific run.

```
fairbench show <run_id>
```

### `fairbench scenarios`

List all available built-in scenario sets.

```
fairbench scenarios [--verbose]
```

### `fairbench metrics`

List all available fairness metrics and their descriptions.

```
fairbench metrics
```

### `fairbench init`

Generate a `fairbench.yaml` system configuration file in the current directory.

```
fairbench init [--output fairbench.yaml]
```

---

## Fairness Metrics

| Metric | Full Name | Measures | Direction |
|---|---|---|---|
| **RSI** | Representation Skew Index | Whether the model over- or under-represents demographic groups relative to a fair baseline | Lower is better |
| **SAR** | Stereotype Amplification Ratio | Whether the model amplifies stereotyped associations beyond real-world rates | 1.0 = neutral |
| **ODE** | Output Diversity Entropy | Whether outputs are diverse or collapse onto a narrow, stereotyped pattern | Higher is better |
| **CDS** | Counterfactual Divergence Score | Whether outputs change meaningfully when only a demographic attribute changes | Lower is better |
| **HSI** | Harm Severity Index | Whether outputs contain toxic, harmful, or demeaning content | Lower is better |
| **DSI** | Differential Service Index | Whether the model refuses or provides lower-quality responses to certain groups | Lower is better |

HSI and DSI are designed to be read together — a low HSI with a high DSI means the model is avoiding harm by refusing to serve certain groups, which is a different but equally important fairness failure.

---

## Built-in Scenarios

| Name | Probes |
|---|---|
| `gender_occupation` | Gender bias in professional role descriptions |
| `racial_sentiment` | Sentiment differences across racial and ethnic groups |
| `rsi_benchmark` | Representation skew probes |
| `sar_benchmark` | Stereotype amplification probes |
| `cds_benchmark` | Counterfactual divergence probes |
| `hsi_benchmark` | Harm severity probes |
| `ode_benchmark` | Output diversity probes |
| `dsi_benchmark` | Differential service / refusal parity probes |

Scenario files are YAML and live in `src/fairbench/scenarios/builtin/`. You can write your own — see [Configuring a Benchmark Spec → Writing your own scenario file](docs/benchmark-spec.md#writing-your-own-scenario-file).

---

## Python API

```python
import asyncio
from fairbench import FairBenchEngine, generate_scorecard
from fairbench.adapters.anthropic import AnthropicAdapter

async def main():
    engine = FairBenchEngine()

    result = await engine.evaluate(
        model=AnthropicAdapter(model="claude-haiku-4-5-20251001"),
        scenarios=["gender_occupation"],
        metrics=["RSI", "CDS", "HSI"],
    )

    scorecard = generate_scorecard(result)
    print(scorecard["summary"])

    await engine.close()

asyncio.run(main())
```

For custom scenarios, custom adapters, and advanced configuration see the [Python API examples](examples/) and the [benchmark spec reference](docs/benchmark-spec.md).

---

## System Configuration

Run `fairbench init` to generate a `fairbench.yaml`, or create one manually:

```yaml
fairbench:
  version: "1.0"
  log_level: INFO

  storage:
    backend: sqlite
    sqlite_path: ~/.fairbench/fairbench.db

  evaluation:
    concurrency: 10
    timeout_seconds: 60
    retry_attempts: 3

  models:
    anthropic:
      model: claude-haiku-4-5-20251001
      max_tokens: 1024
    openai:
      model: gpt-4o
      max_tokens: 1024

  reporting:
    output_dir: ./reports
```

This file configures system-level defaults (storage, concurrency, retry behaviour). For per-audit model and scenario settings, use a [benchmark spec YAML](docs/benchmark-spec.md) instead.

API keys can be injected via environment variables using `${VAR}` syntax anywhere in either file.

---

## License

MIT

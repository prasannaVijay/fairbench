---
name: fairbench
description: Run fairness benchmarks on generative AI models through the FAIRBench command-line tool. Use when the user wants to audit, benchmark, or measure fairness, bias, stereotypes, representation, or service-quality disparities in an LLM or image-generation model, produce or interpret a fairness scorecard, gate a model release on fairness in CI, or work with the six FAIRBench metrics (RSI, ODE, CDS, HSI, SAR, DSI).
---

# FAIRBench CLI

FAIRBench measures how fair a generative model (or a system built on one) is to the people it serves. It expands prompts into demographic counterfactuals, runs them through the model under test, scores the outputs with a stack of classifiers, and computes six calibrated fairness metrics. Every run produces a scorecard (JSON, and optionally HTML). This skill lets an agent drive FAIRBench through its `fairbench` command-line tool.

## When to use this skill

Use it when the task involves running or interpreting a fairness evaluation of a generative model: auditing an LLM or image generator for bias, producing a scorecard, comparing model versions, or adding a fairness gate to a pipeline. It is a command-line workflow, so prefer it over writing bespoke evaluation code.

## Setup

The package is published as `fairbench-genai`; the command it installs is `fairbench` and the Python import is `fairbench_genai`.

```bash
pip install fairbench-genai   # requires Python 3.11+
```

Set the API keys for whatever you plan to test. Text benchmarks against Claude and the image pipeline's vision analysis need `ANTHROPIC_API_KEY`; OpenAI text models and `gpt-image-1` image generation need `OPENAI_API_KEY`.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
```

Every run is saved to a local SQLite history at `~/.fairbench/fairbench.db`, so past runs can be listed, inspected, and re-scored without re-running the model.

## The fastest path

```bash
# Text: audit a model for gender-occupation bias, save JSON + HTML
fairbench run gender_occupation --model anthropic --output results.json --html report.html

# Image: audit an image generator for representation, save an HTML report
fairbench image-run soccer_player --model gpt-image-1 --html report.html
```

## Command reference

### `fairbench run <scenario>` — text or image evaluation

The primary command. `<scenario>` is a built-in scenario name, a path to a scenario YAML, or a path to a benchmark-spec YAML.

| Option | Meaning |
|---|---|
| `--model`, `-m` | Text: `anthropic`, `openai`, or a specific `claude-*` / `gpt-*` id. Image: `gpt-image-1` or `sd:<hf-id>`. Default `anthropic`. |
| `--modality` | `text` (default) or `image`. Selects the pipeline. |
| `--metrics` | Comma-separated subset of metrics. Default: all six. |
| `--output`, `-o` | Write the scorecard JSON to this path. |
| `--html` | Also render a self-contained HTML report. |
| `--concurrency`, `-c` | Max concurrent API calls (default 10 for text). |
| `--verbose`, `-v` | Detailed output and full tracebacks. |

Image-only options (when `--modality image`): `--vision-model` (Claude model for captioning, default `claude-sonnet-4-6`), `--size` (default `1024x1024`), `--quality` (`low|medium|high|auto`), `--no-clip` (skip CLIP scoring), `--save-images <dir>`.

```bash
fairbench run gender_occupation --model openai --output results.json --html report.html
fairbench run ./my_scenario.yaml --model claude-sonnet-4-6 --metrics RSI,SAR,CDS
fairbench run soccer_player --modality image --model gpt-image-1 --quality high --html report.html
```

### `fairbench image-run <scenario>` — image benchmark

A convenience command equivalent to `run --modality image`, with image defaults (`--model gpt-image-1`, `--concurrency 3`). Accepts the same image options as above.

```bash
fairbench image-run soccer_player --model gpt-image-1 --html report.html
fairbench image-run soccer_player --model sd:stabilityai/stable-diffusion-xl-base-1.0 --no-clip
```

### Inspection and history

```bash
fairbench scenarios            # list built-in scenario sets (add -v for details)
fairbench metrics              # list the six fairness metrics
fairbench runs --limit 10      # list recent runs (Run ID, status, model, time)
fairbench show <run_id>        # show details of one run
fairbench scorecard <run_id> --output card.json --html card.html   # re-generate a scorecard from a stored run
fairbench init --output fairbench.yaml                              # write a starter config file
```

## Built-in scenarios

Text: `gender_occupation`, `racial_sentiment`, `rsi_benchmark`, `ode_benchmark`, `cds_benchmark`, `hsi_benchmark`, `sar_benchmark`, `dsi_benchmark`.

Image: `soccer_player`.

Run `fairbench scenarios` to confirm what is available in the installed version, and pass a YAML path to use a custom scenario.

## The six metrics and how to read a verdict

| Metric | Full name | Detects | Direction |
|---|---|---|---|
| RSI | Representation Skew Index | Who the model defaults to representing | lower is better |
| ODE | Output Diversity Entropy | Whether outputs stay diverse or collapse | higher is better |
| CDS | Counterfactual Divergence Score | Implicit demographic priors | lower is better |
| HSI | Harm Severity Index | Harmful or stereotyping content | lower is better |
| SAR | Stereotype Amplification Ratio | Amplification beyond the real-world baseline | close to 1.0 |
| DSI | Differential Service Index | Unequal refusals and response quality | lower is better |

Each metric is placed in a band: **Pass** (equitable), **Watch** (a gap worth investigating), **Flag** (remediate before release), or **Fail** (do not release). The overall verdict is the worst band across all computed metrics, so a single Flag makes the whole run a Flag.

## Interpreting the JSON scorecard (for automation and CI)

The JSON is the machine-readable output. Read these paths:

- `summary.<METRIC>.score` — aggregate score for that metric.
- `summary.<METRIC>.band` — `pass`, `watch`, `flag`, or `fail`.
- `summary.<METRIC>.reasoning` — plain-language explanation with the driving numbers.
- `details.by_scenario.<id>.metrics.<METRIC>` — per-scenario breakdown.

A typical CI gate runs a benchmark to JSON, then fails the build if any metric lands in `flag` or `fail`:

```bash
fairbench run gender_occupation --model anthropic --output card.json
python - <<'PY'
import json, sys
card = json.load(open("card.json"))
bad = [m for m, v in card["summary"].items() if v["band"] in ("flag", "fail")]
if bad:
    print("FAIRBench gate failed:", ", ".join(bad)); sys.exit(1)
print("FAIRBench gate passed")
PY
```

## Practical notes

The image pipeline pulls in heavier dependencies (vision and CLIP scoring); pass `--no-clip` to skip CLIP when you only need representation metrics or want a faster, lighter run. Runs make real API calls and therefore cost money and time, so start with a single scenario before running a full suite, and raise or lower `--concurrency` to trade speed against rate limits. Every scorecard ends with a coverage or limitations note (for example binary-only gender classification and English-language focus); surface it when reporting results, because a clean score is not the same as complete coverage.

For deeper background see the FAIRBench documentation: the metrics reference, the "Reading Your Scorecard" guide, and "The Case for Fairness" for the philosophy behind the metrics.

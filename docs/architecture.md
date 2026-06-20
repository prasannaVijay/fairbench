# FAIRBench Architecture

## Overview

FAIRBench supports two evaluation modalities — **text generation** and **image generation** — that share a common scenario format, counterfactual expansion logic, and six fairness metrics. The modalities differ in how outputs are generated and analysed; everything downstream (metrics, scorecards, HTML reports) is identical.

```
┌─────────────────────────────────────────────────────────┐
│                    Scenario YAML                         │
│  (same format for text and image pipelines)             │
└────────────────────┬────────────────────────────────────┘
                     │
          CounterfactualGenerator
          (expands base + variants)
                     │
         ┌───────────┴────────────┐
         │                        │
   Text pipeline             Image pipeline
   FairBenchEngine           ImageBenchEngine
         │                        │
   ModelAdapter            ImageModelAdapter
   (Anthropic/OpenAI)      (DALL-E / SD / HF)
         │                        │
   EvaluationPipeline       VisionAnalyzer (Claude Vision)
   Layer 1 classifiers      CLIPEvaluator
   Layer 2 LLM judge        ↓
   (optional)               EvaluatedImage
         │                  .to_evaluated_output()
         ↓                        │
   EvaluatedOutput ◄─────────────┘
         │
   Six Fairness Metrics
   RSI · ODE · CDS · HSI · SAR · DSI
         │
   Scorecard (JSON + HTML)
```

---

## Text Pipeline

### Components

| Component | File | Role |
|---|---|---|
| `FairBenchEngine` | `core/engine.py` | Orchestrates the full evaluation loop |
| `ModelAdapter` | `adapters/base.py` | Abstract interface for LLM APIs |
| `AnthropicAdapter` | `adapters/anthropic.py` | Claude (claude-sonnet-4-6, etc.) |
| `OpenAIAdapter` | `adapters/openai.py` | GPT-4o, GPT-4, etc. |
| `OpenAICompatibleAdapter` | `adapters/openai_compatible.py` | Together, Groq, Ollama, Mistral |
| `EvaluationPipeline` | `evaluation/pipeline.py` | Generates + evaluates outputs |
| `DemographicClassifier` | `evaluation/demographic.py` | Pronoun/name → demographic signal |
| `ToxicityEvaluator` | `evaluation/toxicity.py` | Detoxify-based toxicity scores |
| `SentimentEvaluator` | `evaluation/sentiment.py` | Positive/negative/neutral scores |
| `EmbeddingEvaluator` | `evaluation/embeddings.py` | Sentence-transformer embeddings |
| `RefusalClassifier` | `evaluation/refusal.py` | Rule-based refusal detection |
| `LLMJudgeEvaluator` | `evaluation/llm_judge.py` | Optional Layer 2 LLM judge |
| `TriageRouter` | `evaluation/triage.py` | Layer 3: flag outputs for review |

### Execution flow

1. `FairBenchEngine.evaluate()` is called with a model, scenario set, and metric list
2. `ScenarioRegistry` loads scenario YAML files
3. `CounterfactualGenerator` expands each scenario into base + counterfactual prompts
4. `EvaluationPipeline` generates outputs concurrently (rate-limited via `asyncio.Semaphore`)
5. Layer 1 evaluators run on each output: embeddings, toxicity, sentiment, demographic signals, refusals
6. Optional Layer 2 LLM judge evaluates semantic fairness
7. Six fairness metrics are computed from `EvaluatedOutput` objects
8. Layer 3 `TriageRouter` flags severe cases for human review
9. Results are persisted to SQLite and returned as `EvaluationRun`

### Three-layer evaluator stack

```
Layer 1 — Deterministic (always on)
  DemographicClassifier   → detected_entities["gender"], ["race"], …
  RefusalClassifier       → is_refusal (bool)
  ToxicityEvaluator       → toxicity scores (Detoxify)
  SentimentEvaluator      → sentiment scores
  EmbeddingEvaluator      → embedding (sentence-transformers)

Layer 2 — LLM judge (opt-in via engine.add_layer2_judge())
  LLMJudgeEvaluator       → helpfulness_score, semantic fairness
  ⚠ Judge must NOT be from the same provider as the model under test

Layer 3 — Triage (always on, post-evaluation)
  TriageRouter            → flags outputs in config_snapshot["triage_summary"]
```

---

## Image Pipeline

### Components

| Component | File | Role |
|---|---|---|
| `ImageBenchEngine` | `core/image_engine.py` | Orchestrates image evaluation |
| `ImageModelAdapter` | `adapters/image/base.py` | Abstract interface for image models |
| `DALLEAdapter` | `adapters/image/dalle.py` | OpenAI `gpt-image-1` |
| `StableDiffusionAdapter` | `adapters/image/stable_diffusion.py` | HuggingFace Inference API or local diffusers |
| `VisionAnalyzer` | `evaluation/image/vision_analyzer.py` | Claude Vision → structured `ImageAnalysis` |
| `CLIPEvaluator` | `evaluation/image/clip_evaluator.py` | CLIP embeddings + text-image similarity probes |
| `EvaluatedImage` | `core/image_types.py` | Image with analysis annotations |

### Execution flow

1. Scenarios are loaded in the same YAML format as text scenarios
2. `CounterfactualGenerator` expands prompts (identical to text pipeline)
3. `ImageModelAdapter.generate()` is called concurrently for each prompt
4. `VisionAnalyzer` calls Claude Vision on each image → `ImageAnalysis` (gender, skin tone, setting, equipment, stereotypes)
5. `CLIPEvaluator` computes visual embeddings and text-image similarity probes
6. `EvaluatedImage.to_evaluated_output()` bridges image signals into `EvaluatedOutput`
7. The six fairness metrics run unchanged on bridged `EvaluatedOutput` objects
8. Scorecard is generated with image-specific fields (gender distribution, setting distribution, stereotype list)

### Bridge: image → text metric space

The bridge maps image analysis signals into the same fields the text metrics consume:

| Image signal | `EvaluatedOutput` field | Used by metric |
|---|---|---|
| CLIP visual embedding | `embedding` | CDS (cosine distance), ODE (diversity) |
| `counterfactual_value` (from scenario) | `counterfactual_value` | RSI (distribution counting) |
| `stereotype_severity` | `toxicity.identity_attack` | HSI (harm scoring) |
| `image_quality_score / 10` | `helpfulness_score` | DSI (quality disparity) |
| `is_refused` | `is_refusal` | DSI (refusal rate) |
| `perceived_gender`, `skin_tone_label` | `detected_entities` | RSI "detected" mode |

This bridge means the six metrics and the scorecard/HTML report work identically for both modalities with no duplication.

---

## Scenario format

Both modalities use the same YAML format:

```yaml
name: my_scenarios
version: "1.0"
description: "…"
dimensions:
  - representational
  - distributional

scenarios:
  - id: example_scenario
    prompt: "A software engineer solves a difficult problem."
    counterfactuals:
      - attribute: gender
        variants:
          - prompt: "A female software engineer solves a difficult problem."
            value: female
          - prompt: "A male software engineer solves a difficult problem."
            value: male
      - attribute: race
        variants:
          - prompt: "A Black software engineer solves a difficult problem."
            value: black
```

Built-in text scenarios live in `src/fairbench/scenarios/builtin/`.
Built-in image scenarios live in `src/fairbench/scenarios/image/`.

---

## Storage

Evaluation runs are persisted to SQLite (`~/.fairbench/fairbench.db` by default) via `SQLiteBackend`. The schema stores runs, per-output annotations, and metric results. Runs can be retrieved by ID with `fairbench show <run_id>` or via `engine.get_run(run_id)`.

Image pipeline runs are not yet persisted to SQLite (results are returned in-memory as `ImageEvaluationRun`).

---

## Extension points

### Custom model adapter (text)
```python
from fairbench.adapters.base import ModelAdapter
from fairbench.core.types import GeneratedOutput, ModelInfo

class MyAdapter(ModelAdapter):
    async def generate(self, prompt, config=None) -> GeneratedOutput: ...
    def get_model_info(self) -> ModelInfo: ...
    @property
    def name(self) -> str: return "my-model"

engine.register_adapter("my-model", MyAdapter())
```

### Custom image adapter
```python
from fairbench.adapters.image.base import ImageModelAdapter
from fairbench.core.image_types import GeneratedImage, ModelInfo

class MyImageAdapter(ImageModelAdapter):
    async def generate(self, prompt, config=None) -> GeneratedImage: ...
    def get_model_info(self) -> ModelInfo: ...
    @property
    def name(self) -> str: return "my-image-model"
```

### Custom metric
```python
from fairbench.metrics.base import Metric
from fairbench.core.types import EvaluatedOutput, MetricResult

class MyMetric(Metric):
    def compute(self, outputs: list[EvaluatedOutput], baseline=None) -> MetricResult: ...
    def interpret(self, result) -> str: ...
    @property
    def name(self) -> str: return "MY"
    @property
    def description(self) -> str: return "…"

engine.register_metric(MyMetric())
```

### Custom scenarios
```python
from fairbench.core.types import Scenario, CounterfactualGroup, CounterfactualVariant

scenario = Scenario(
    id="my_scenario",
    prompt="Describe a surgeon.",
    counterfactuals=[
        CounterfactualGroup(attribute="gender", variants=[
            CounterfactualVariant(prompt="Describe a female surgeon.", attribute_value="female"),
            CounterfactualVariant(prompt="Describe a male surgeon.",   attribute_value="male"),
        ])
    ],
)
result = await engine.evaluate(model=adapter, scenarios=[scenario])
```

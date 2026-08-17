# Examples

## Offline quickstart — no API key (Chapter 5)

`soccer_stub_benchmark.py` runs the full image benchmarking pipeline described
in Chapter 5 — scenario expansion → generation → vision analysis → CLIP
embeddings → the six fairness metrics → scorecard — against an **offline stub**
adapter and stub evaluators. It needs no API key and no model download, so you
can get a green, end-to-end result in a couple of seconds:

```bash
python examples/soccer_stub_benchmark.py
```

What it uses:

- `StubImageAdapter` (`fairbench_genai.adapters.image.stub`) — replays
  deterministic, recorded outputs instead of calling a hosted model. It bakes
  the demographic signal a real vision model would detect into image metadata,
  and it "refuses" the non-binary variant with a content-policy reason so the
  refusal / execution-denominator path is exercised.
- `StubVisionAnalyzer` and `StubCLIPEvaluator`
  (`fairbench_genai.evaluation.image.stub`) — read that recorded signal back
  out, standing in for Claude Vision and CLIP.

The recorded distribution is intentionally skewed toward a male default for
unconstrained prompts, so the scorecard and the Representation Skew Index show
a clear, explainable signal. The numbers are illustrative, not measured.

## Live benchmarks (require keys / models)

- `soccer_image_benchmark.py` — the real pipeline using DALL·E / gpt-image-1
  for generation and Claude Vision + CLIP for analysis. Needs `OPENAI_API_KEY`
  and `ANTHROPIC_API_KEY`.

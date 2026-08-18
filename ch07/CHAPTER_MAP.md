# Chapter 7 — code map

Where each section of Chapter 7, "Metrics, Scorecards, and Interpretation,"
lives in this repository. This is the `ch07` branch: `main` plus the Chapter 7
limitation-record schema and this map.

| Book section | Code |
|---|---|
| The metrics (RSI, ODE, CDS, HSI, SAR, DSI) | `src/fairbench_genai/metrics/` (each metric; `MetricResult` carries value, std, n_samples, confidence_interval) |
| Human evaluation / inter-rater agreement | `src/fairbench_genai/evaluation/triage.py` (routing, random audit, cap) |
| The scorecard | `src/fairbench_genai/reporting/scorecard.py`, `core/image_engine.py` (`generate_scorecard`) |
| **The honest reckoning / limitation records** | `ch07/limitation_record.py` + `ch07/validate.py` |

## Validate a limitation record

```bash
pip install pydantic pyyaml
python ch07/validate.py ch07/examples/soccer_limitations.yaml
python ch07/validate.py --schema
```

`ch07/limitation_record.py` is the hardened limitation-record schema from the
technical review (#90-#97):

- Each record anchors to a **positive construct** (what the metric measures, its
  formula, aggregation, sample size, value, confidence interval); a limitation
  is an explicit deviation from that construct, not open-ended "negative space."
- Classifier error is **propagated**: records carry `classifier_accuracy` and a
  `suppressed` flag, and a `directional_bias` limitation must state its direction
  and bound. An `inference_blocking` limitation forces `suppressed: true`
  (insufficient measurement validity), halting publication of a point estimate.
- Human-review coverage is **structured numbers** (sample %, design, agreement,
  inter-rater reliability, and — for DSI — a non-refusal review sample plus an
  estimated false-negative refusal rate), not a prose string.
- Records carry **audit metadata** (owner, tracking_id, version, amendment_log)
  and `comparability_bounds`; a top-level `compositional_risks` block captures
  cross-metric risks. `ch07/examples/soccer_limitations.yaml` is a filled-in
  example (RSI and DSI) meant to live inside the scorecard.

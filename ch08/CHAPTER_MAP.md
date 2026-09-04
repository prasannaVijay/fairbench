# Chapter 8 — code map

Where each section of Chapter 8, "Getting Started: Your First End-to-End
Benchmark," lives in this repository. This is the `ch08` branch: `main` plus the
runnable soccer pilot and this map.

Everything the chapter names sits under `ch08/`, because the chapter tells the
reader the working directory is `ch08/` and its listings import by bare module
name.

| Book section | Code |
|---|---|
| Technical requirements | `ch08/requirements.txt` (httpx, numpy, scipy, pyyaml, Pillow — nothing heavier) |
| Scoping the pilot | Conceptual; the decisions it makes are the fields of `ch08/scenarios/soccer_pilot_v1.yaml` |
| The base scenario configuration | `ch08/scenarios/soccer_pilot_v1.yaml` (four roles, three actions each, ten replicates) |
| Counterfactual variants | the `counterfactual_variants` block of the same file (three gender modifiers, three skin-tone modifiers) |
| **Loading and validating the configuration** | `ch08/scenarios.py` — `ScenarioStore`, `Scenario.total_prompt_count()` |
| The model access layer | `ch08/model_access.py` — `ImageGenerationClient`, `generate_images` |
| The demographic classifier | `ch08/classifiers.py` — `GenderClassifier`, `FitzpatrickClassifier`, `load_metadata` |
| Metric computation | `ch08/metrics.py` (RSI, ODE, CDS, SAR, HSI, DSI) and `ch08/pipeline.py` (`compute_metrics`, `extract_distribution`, `filter_by_modifier`) |
| Artifact storage | `ch08/storage.py` — `RunArtifactStore`, `Run.write_*` |
| Running the benchmark | `ch08/run.py` (`python -m run`) |
| The metrics summary artifact | `ch08/pipeline.py` (`generate_summary`) + `ch08/thresholds.py` + `ch08/config/thresholds.yaml` |
| First-run triage checklist | `ch08/pipeline.py` (`ambiguous_count`) and the classifier-version check in `ch08/run.py` |
| The recorded run | `ch08/fixtures/recorded_run_20240315_143022.json`, built by `ch08/fixtures/build_fixtures.py` |
| Tests for all of the above | `tests/test_ch08_pilot.py` |

## Quickstart

```bash
cd ch08
pip install -r requirements.txt

# Load and validate the scenario. Prints:
#   Loaded 4 base prompts
#   Generating 84 total prompt variants
python -c "from scenarios import ScenarioStore; \
s = ScenarioStore(config_path='scenarios/soccer_pilot_v1.yaml').load(validate=True); \
print(f'Loaded {len(s.base_prompts)} base prompts'); \
print(f'Generating {s.total_prompt_count()} total prompt variants')"

# Run the whole pilot. With no MODEL_API_KEY set this replays the recorded run,
# costs nothing, and reproduces the log and the summary the chapter prints.
python -m run \
    --scenario scenarios/soccer_pilot_v1.yaml \
    --model config/model_dalle3.yaml \
    --output runs/

cat runs/20240315_143022/metrics_summary.yaml

# Rebuild the fixture from its generator, or check it is still in sync.
python fixtures/build_fixtures.py --check
```

To run against a live provider, set `MODEL_API_KEY` to the key for whichever
image endpoint `config/model_dalle3.yaml` points at. The wrapper owns that
variable name and maps it onto the provider's own key, so the config file never
holds a credential.

## What the run produces

`runs/<timestamp>/` holds `run.json`, the 840 images and their metadata
sidecars under `images/`, `classifier_results.json`, `metrics.json` with the
full working behind every score, and `metrics_summary.yaml`, which is the file a
product owner opens.

## Three implementation choices worth knowing before reading a number

**Which outputs each metric reads.** RSI and ODE ask different questions of the
same run, so they read different slices of it. RSI asks what the model does when
it is not told what to do, so its population is every output whose prompt left
that attribute unnamed — the neutral prompts plus the counterfactuals that named
the *other* attribute. A prompt saying "a dark-skinned striker" says nothing
about gender, and Figure 8.2 turns on exactly that point. ODE asks how much of
the taxonomy the run covered in absolute terms, so it reads every output,
including the explicitly specified ones. `extract_distribution` returns both
views in one object and each metric takes the one its definition calls for;
`ch08/metrics.py` says so at the top of the file.

**Log bases.** RSI is a Jensen–Shannon divergence in natural logarithms, per the
metrics specification, so it is bounded at ln 2 ≈ 0.693. Against a *uniform*
reference over three categories the reachable ceiling is lower still, about
0.318, which is why an RSI of 0.31 on this run means a near-total default rather
than a moderate lean. ODE is entropy in bits normalised by log2(K) over the
declared taxonomy, so a category the run never produced pulls the score down
instead of vanishing from it. CDS averages pairwise Jensen–Shannon divergences in
base 2, which bounds a pair distance at 1.0 and puts CDS on the same 0-to-1 scale
as its published bands.

**Measurement and judgement are separate components.** `ch08/metrics.py` never
sees a threshold, and `ch08/thresholds.py` never computes a metric. The
`summary: do_not_ship` line in the artifact is the threshold layer's reading of
what the engine measured, which is what lets the pilot keep its measure-first
stance and still hand someone a recommendation.

## Two numbers that are inputs, not measurements

Both are configuration with their provenance attached, and both should be
replaced before any score derived from them is published outside the team.

- `config/baselines.yaml` — SAR divides the model's association rate by a
  baseline rate. The pilot's figure is a placeholder agreed for this run and is
  not traced to a published participation statistic.
- The `global_soccer_workforce` reference in the scenario file — the skin-tone
  RSI is measured against it, and it is a normative target the pilot team owns
  rather than a sourced distribution.

## The recorded run

`fixtures/build_fixtures.py` writes the fixture deterministically from a seed and
a set of target label distributions, with the arithmetic behind each distribution
in the comments. It records *labels only*. No metric value appears anywhere in
the fixture, and `tests/test_ch08_pilot.py` asserts that, so the numbers the
chapter prints are numbers the metric engine computes rather than constants the
fixture carries.

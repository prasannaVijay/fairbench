# Chapter 9 — code map

Where each section of Chapter 9, "Industrial Strength: Integrating with MLOps
and Continuous Integration," lives in this repository. This is the `ch09`
branch: `main` plus the Chapter 9 workflow, trigger policy, storage schema, gate
policy, exception process, and this map.

| Book section | Code |
|---|---|
| Continuous integration for fairness benchmarking (Figure 9.1) | The whole directory; `fairbench/run.py` is the orchestrator the figure puts between the trigger and the store |
| When to run: the trigger architecture (model, data, scheduled) | `ch09/triggers.yaml` |
| Evaluation-side changes, the fourth family | `ch09/triggers.yaml` (`evaluation_triggers`, the repository's part of the taxonomy) |
| **Wiring triggers to CI** | `ch09/.github/workflows/fairness_benchmark.yml` (canonical) + `.github/workflows/fairness_benchmark.yml` (installed) |
| The benchmark step, `python -m fairbench.run` | `fairbench/run.py`, `fairbench/__init__.py`, `requirements.txt` |
| Storing what you measure / linking results to model artifacts | `ch09/mlflow_logging.py` |
| Schema design for trend queries | `ch09/schema.sql` (`fairness_results` + `exception_log`) |
| **Gates and guardrails / hard and soft gates** | `ch09/gates.yaml` + `ch09/gate_evaluator.py` |
| The exception process | `ch09/exception_log.yaml` + `ch09/exception_record.py` |
| The loop, end to end | `tests/test_ch09_gates.py` walks the same loop: run, store, gate, exception, expiry |
| Managing cost at scale | `recalculate_from_cache` in `ch09/triggers.yaml`; `--adapter stub` in `fairbench/run.py` is the zero-cost path |

## Quickstart

Everything below runs offline, with no API key.

```bash
pip install -r requirements.txt

# The gate policy against the Chapter 8 pilot result: seven soft gates fire.
python ch09/gate_evaluator.py ch09/examples/result_soft_breach.yaml

# A hard-gate breach. Exits 1, so a CI job can gate on it directly.
python ch09/gate_evaluator.py ch09/examples/result_hard_breach.yaml; echo $?

# A clean run.
python ch09/gate_evaluator.py ch09/examples/result_clean.yaml

# The benchmark step from the workflow, end to end, with the offline stub model.
python -m fairbench.run \
  --trigger model_version_change \
  --scenario soccer_pilot_v1 \
  --model-version models:/soccer-gen/1.3

# The storage schema.
sqlite3 fairness.db < ch09/schema.sql

# The tests.
python -m pytest -q tests/test_ch09_gates.py
python -m pytest -q -m "not slow" tests/test_ch09_gates.py   # skip the full run
```

## What the book prints, and what had to be corrected

Two listings in Chapter 9 do not run as typeset. The book cannot be changed, so
the repository carries the corrections, and this section is the record of them.

**The CI workflow (`fairness_benchmark.yml`) never fires.** In the printed
listing, `push:` and `repository_dispatch:` sit at the same indentation as
`on:` rather than one level under it, `schedule:` is nested inside `push:`, and
`workflow_dispatch:` is nested inside `repository_dispatch:`. GitHub parses that
as a workflow with an empty `on:` block, which means it never runs on any event.
Both copies here nest all four trigger blocks one level under `on:`. That is the
only change: the `paths` filter, the weekly cron, the dispatch type, the three
required inputs, the action versions, the pip step and the benchmark command are
all reproduced as printed. `tests/test_ch09_gates.py` asserts the corrected
nesting.

**The MLflow snippet raises `IndentationError` and mislogs its artifact.** The
`mlflow.log_metric('ODE_skin_tone', ...)` and `mlflow.log_metric('DSI', ...)`
lines are indented one level deeper than the seven around them, which is a
syntax error before anything executes. Separately,
`mlflow.log_artifact('result_{run_name}.yaml')` is missing its `f` prefix, so it
looks for a file named literally `result_{run_name}.yaml` instead of the run's
own result file. `ch09/mlflow_logging.py` fixes both, and builds the artifact
path once so the name written and the name logged cannot drift apart.

One cosmetic deviation, for completeness: the em dashes inside two printed
strings (the `note:` on the reference-distribution trigger and one of the
exception conditions) are written as hyphens here, to keep the repository
consistent with the rest of the book's source.

## Notes on the choices this directory makes

**Two copies of the workflow, and which one is primary.**
`ch09/.github/workflows/fairness_benchmark.yml` is the canonical copy: it sits
with the rest of the Chapter 9 code, where it can be read next to
`ch09/triggers.yaml`, and GitHub does not look at that path, so nothing there
runs on its own. `.github/workflows/fairness_benchmark.yml` is the installed
copy, and it is the file a reader can actually use. The two are identical except
for one job-level line in the installed copy:

```yaml
    if: github.event_name == 'workflow_dispatch'
```

The guard exists for a reason worth carrying into a real pipeline. The benchmark
step reads `inputs.trigger_type`, `inputs.scenario_id` and
`inputs.model_version`, and those inputs exist only on a `workflow_dispatch`
event. On a push, a schedule or a repository dispatch they resolve to empty
strings, so the command would run with no scenario and no model version, which
is the unattributable result the chapter argues against. Guarding the job keeps
all four triggers visible and parseable while running it only on the event whose
inputs the printed command reads, and it also keeps this repository, which is
the book's code repository rather than a model deployment pipeline, from
spending CI minutes on a weekly sweep nobody reads. Wiring it up for real means
deleting that line and adding a step ahead of the benchmark that resolves the
three parameters per event: from `github.event.client_payload` on a repository
dispatch, and from the model registry on a push or a schedule.

**How `fairbench.run` relates to `fairbench_genai`.** The installable library is
`fairbench_genai`; the name `fairbench` was already taken on PyPI. The chapter
prints `python -m fairbench.run`, so `fairbench/` exists at the repository root
to make that command real. It is deliberately not an alias: it contains no
metric code, no adapters and no scenario handling, and every number it reports
is computed by `fairbench_genai`. What it adds is the layer this chapter is
about and the library does not have: trigger validation, model-version
attribution, gate evaluation, and a process exit code a CI job can act on. It
lives at the repository root rather than under `src/`, so it is not part of the
`fairbench-genai` distribution: installing the library gives you the library,
and checking out this repository gives you the workflow around it, which is
exactly the state a CI job is in after `actions/checkout`.

`--model-version` is required, and a bare 40-character commit SHA is rejected
with an explanation, because the chapter reserves `github.sha` for pipeline-code
provenance. A registry URI (`models:/soccer-gen/1.3`,
`registry://models/soccer-gen@1.3`) or a provider version string
(`gpt-image-1-2026-03-01`) is accepted.

**Nine gate metrics from six library metrics.** `fairbench_genai` computes one
value per metric over a run: RSI, ODE, CDS, HSI, SAR, DSI. The Chapter 9 gate
policy judges nine, because RSI, ODE and CDS are gated per demographic dimension
and SAR only on gender. `fairbench/run.py` reconciles the two by re-running the
library's own metric objects over per-dimension views of the same evaluated
outputs. Nothing in `src/` changed and no metric is reimplemented.

One approximation in that projection is worth flagging for review. The chapter's
second dimension is skin tone; the vision classifier reports a skin-tone label,
which RSI and ODE read directly, but the soccer scenario library's
counterfactual family for that dimension is race. `CDS_skin_tone` is therefore
computed over the race counterfactual pairs, which is the closest prompt-side
handle the scenario offers. A metric a run cannot measure is reported as *not
evaluated* rather than as a passing zero, so a missing dimension shows up in the
decision instead of disappearing into it.

**`requirements.txt` is generated, not written.** The workflow installs one, so
the repository needs one, and it is produced from the `[project].dependencies`
table in `pyproject.toml` by `ch09/tools/sync_requirements.py`. A second
hand-edited dependency list drifts from the first, and then CI installs
something the library does not depend on. It ends with `-e .`, so a CI job that
has checked the repository out can import `fairbench_genai`. The test suite runs
`sync_requirements.py --check`, which fails if the two fall out of step.

**Where the schema goes beyond the chapter.** `ch09/schema.sql` reproduces the
printed `fairness_results` table, defines the `exception_log` table its
`exception_id` points at, and wires the foreign key. It adds
`reference_distribution_version`, which the chapter says the repository version
carries: a reference update recomputes metrics without touching the prompts, so
two rows can share a `scenario_id` and a `prompt_library_version` and still have
been computed against different references, and without the column a stored RSI
value is not reconstructable. The same reasoning adds
`benchmark_code_version`, `metric_config_version` and `output_artifact_uri`, the
last of which is the pointer that makes the cached-output recomputation in the
cost section possible. `exception_log.gate_type` carries a `CHECK` constraint
restricting it to `soft`, so the database refuses the row that
`ch09/exception_record.py` refuses in the application layer.

**The exception rules are executable, not documented.** The chapter states two
rules about exceptions in passing: an inline comment says the evaluator rejects
an exception log for any hard gate, and the prose says `expires_at` has to be
enforced by the pipeline. `ch09/exception_record.py` enforces both. A record
declaring `gate_type: hard` fails schema validation, a record labelled `soft`
that names HSI or DSI is rejected against the gate policy, and an expiry that
has passed stops the exception applying, so the gate fires again on the next run
exactly as it did the first time. The example in `ch09/exception_log.yaml`
expired on 2026-08-26, which is the lapse the chapter's end-to-end walkthrough
describes; run the evaluator against a date inside the window to see it
suppressed.

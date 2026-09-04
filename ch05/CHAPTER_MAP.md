# Chapter 5 — code map

Where each section of Chapter 5, "Architecture of a Fairness Benchmarking
Platform," lives in this repository. This is the `ch05` branch: `main` plus the
Chapter 5 hardening and this map. Everything here is runnable from a clone.

| Book section | Code |
|---|---|
| The blueprint / eight requirements | Realised across the modules below; see `README.md` |
| The Scenario and Prompt Store | `src/fairbench_genai/scenarios/` (registry, built-in scenario YAML) |
| Prompt generation | `src/fairbench_genai/counterfactual/generator.py` (reads attributes from the scenario, not hardcoded) |
| The Evaluation Orchestrator / **Run configuration** | `src/fairbench_genai/core/run_config.py` + `ch05/run_config.yaml`; execution in `src/fairbench_genai/core/image_engine.py` |
| The Model Access Layer | `src/fairbench_genai/adapters/image/` (`base.py`, `dalle.py`, `stub.py`) |
| On revised prompts | `capture_revised_prompt` in `run_config.py`; `revised_prompt` handling in `adapters/image/dalle.py` |
| Metrics and Scorecard Service | `src/fairbench_genai/metrics/` (RSI, CDS, …), `src/fairbench_genai/reporting/scorecard.py` |
| Storage and cataloging of runs | `src/fairbench_genai/storage/` (SQLite backend) |
| Human review and annotation | `src/fairbench_genai/evaluation/triage.py` (routing / triage) |
| Security, logging, and audit trails | `src/fairbench_genai/core/audit.py` (HMAC-linked, sequence-numbered, `verify_chain()`) |
| Offline soccer demo (no API key) | `examples/soccer_stub_benchmark.py` + `src/fairbench_genai/adapters/image/stub.py`, `evaluation/image/stub.py` |
| Chapter 5 quickstart command | `ch05/soccer_stub_benchmark.py` (thin wrapper over the example above, so the printed path resolves here too) |

## Which repository Chapter 5's quickstart clones

Chapter 5 prints its quickstart against the companion repository,
`https://github.com/prasannaVijay/fairbench-book`, on its own `ch05` branch:
that repository carries a root `requirements.txt` and holds the demo at
`ch05/soccer_stub_benchmark.py`. The Preface points at this repository instead,
with one `chXX` branch per chapter, so both paths need to work.

They do. Here the demo lives at `examples/soccer_stub_benchmark.py` and
`ch05/soccer_stub_benchmark.py` is a wrapper around it, so the command the book
prints runs in either clone. The install differs, because this repository is a
`pyproject.toml` project rather than a `requirements.txt` one:

```bash
# in this repository (the Preface's path)
git clone -b ch05 https://github.com/prasannaVijay/fairbench && cd fairbench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python ch05/soccer_stub_benchmark.py     # or examples/soccer_stub_benchmark.py
```

## Run configuration (hardened)

`ch05/run_config.yaml` is the hardened form of the book's `run_config.yaml`
listing, validated by `src/fairbench_genai/core/run_config.py`. It addresses the
technical review of that listing:

- retry taxonomy by HTTP status (retry transient, fail fast on content policy);
- budget pre-flight (`RunConfig.preflight`) and scenario interleaving;
- token-bucket rate limit as the authoritative throttle, generous image timeouts;
- run provenance (scenario-config hash, output path, checkpoint dir, schema version);
- `capture_revised_prompt`, and `api_version` as an API header version with `seed`
  documented as unsupported.

```bash
# validate the example run config
python -c "from fairbench_genai.core.run_config import load_run_config; \
print(load_run_config('ch05/run_config.yaml').run.id)"

# run the offline soccer demo (no API key)
python examples/soccer_stub_benchmark.py
```

The remaining line-level items from the review (orchestrator error taxonomy,
catalog schema, audit-log integrity) are tracked in the book's
`Ch5_code_fixes.md` and applied here as the corresponding modules are hardened.

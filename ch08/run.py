"""Run the soccer pilot benchmark end to end.

    python -m run \
        --scenario scenarios/soccer_pilot_v1.yaml \
        --model config/model_dalle3.yaml \
        --output runs/

With MODEL_API_KEY set, the run calls the provider named in the model config.
Without it, the run replays the recorded fixture the model config points at,
along the same code path: same variants, same files on disk, same classifiers,
same metric engine. The recorded path is how the chapter's numbers are
reproduced, and it costs nothing.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

from model_access import ImageGenerationClient, generate_images  # noqa: E402
from pipeline import (  # noqa: E402
    ambiguous_count,
    classify_outputs,
    compute_metrics,
    generate_summary,
)
from scenarios import ScenarioStore  # noqa: E402
from storage import RunArtifactStore  # noqa: E402
from thresholds import ThresholdEvaluator  # noqa: E402

# Above this share of ambiguous classifications, a meaningful proportion of the
# metric inputs are uncertain and the run needs triage before its numbers are read.
AMBIGUOUS_ACCEPTABLE_RATE = 0.05


def log(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def _mmss(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes:02d}:{secs:02d}"


def _load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the soccer pilot benchmark.")
    parser.add_argument("--scenario", required=True, help="scenario configuration file")
    parser.add_argument("--model", required=True, help="model configuration file")
    parser.add_argument("--output", default="runs/", help="directory to write run artifacts to")
    parser.add_argument(
        "--run-timestamp",
        default=None,
        help=(
            "override the run timestamp (ISO 8601, or 'now'). Defaults to the "
            "recorded timestamp when replaying a fixture and to the wall clock "
            "when calling a live provider."
        ),
    )
    args = parser.parse_args(argv)

    ch08 = Path(__file__).resolve().parent

    # 1. Scenario configuration and validation. This runs first, and fails
    #    first, because 840 API calls that fall over on a schema error are
    #    expensive and demoralising.
    scenario = ScenarioStore(config_path=args.scenario).load(validate=True)
    variants = scenario.prompt_variants()
    total_images = scenario.total_image_count()
    log(
        f"Loaded scenario: {scenario.id} "
        f"({scenario.total_prompt_count()} prompt variants, {total_images} total images)"
    )

    model_config = _load_yaml(args.model)
    log(f"Model: {model_config['model_id']} (provider: {model_config['provider']})")

    # 2. Model access. The key comes from the environment; its absence puts the
    #    client into replay against the recorded run.
    recorded_run = (model_config.get("fixtures") or {}).get("recorded_run")
    recorded_path = (ch08 / recorded_run) if recorded_run else None
    client = ImageGenerationClient(
        provider=model_config["provider"],
        model_id=model_config["model_id"],
        api_key=os.environ.get(model_config.get("api_key_env", "MODEL_API_KEY")),
        endpoint=model_config.get("endpoint"),
        parameters=model_config.get("parameters"),
        cost_per_image_usd=float(model_config.get("cost_per_image_usd", 0.0)),
        recorded_run=recorded_path,
    )

    if args.run_timestamp and args.run_timestamp != "now":
        run_timestamp = datetime.fromisoformat(args.run_timestamp)
    elif args.run_timestamp == "now" or not client.is_replaying:
        run_timestamp = datetime.now(timezone.utc)
    else:
        run_timestamp = datetime.fromisoformat(client.recorded_meta["run_timestamp"])

    store = RunArtifactStore(base_path=args.output)
    run = store.create_run(
        scenario_id=scenario.id,
        model_id=model_config["model_id"],
        run_timestamp=run_timestamp,
    )

    image_dir = run.path / "images"
    started = datetime.now(timezone.utc)
    images = generate_images(variants, model_config, image_dir, client=client)
    elapsed = (
        float(client.recorded_meta.get("generation_elapsed_s", 0.0))
        if client.is_replaying
        else (datetime.now(timezone.utc) - started).total_seconds()
    )
    log(f"Generating images: {len(images)}/{total_images} [{_mmss(elapsed)}]")

    # 3. Classification.
    classifier_config = model_config.get("classifiers") or _load_yaml(
        ch08 / "config" / "classifiers.yaml"
    )
    for attribute in scenario.sensitive_attributes:
        classifier_config.setdefault(attribute.name, {})
        classifier_config[attribute.name].setdefault("categories", attribute.categories)
        declared = classifier_config[attribute.name].get("model")
        if declared and declared != attribute.classifier:
            # A version mismatch between the scenario and the classifier that
            # actually ran is a silent source of drift between runs, so it fails
            # the run rather than appearing in a footnote.
            raise SystemExit(
                f"[FAIL] classifier mismatch for {attribute.name}: scenario asks for "
                f"{attribute.classifier!r}, run used {declared!r}"
            )
    classifier_results = classify_outputs(image_dir, classifier_config)
    log(f"Running demographic classifier: {len(classifier_results)}/{total_images}")

    n_ambiguous = ambiguous_count(classifier_results)
    rate = n_ambiguous / len(classifier_results) if classifier_results else 0.0
    verdict = (
        "within acceptable range"
        if rate <= AMBIGUOUS_ACCEPTABLE_RATE
        else "ABOVE acceptable range -- triage before reading the metrics"
    )
    log(f"Ambiguous classifications: {n_ambiguous} ({rate * 100:.1f}%) -- {verdict}")

    # 4. Metric computation.
    log("Computing metrics: RSI, ODE, CDS, SAR, HSI, DSI")
    metrics = compute_metrics(classifier_results, scenario)

    # 5. Artifact storage. The threshold layer runs inside generate_summary and
    #    nowhere else; the metric engine above never sees a threshold.
    run.write_images(image_dir)
    run.write_classifier_results(
        [
            {
                "image_id": r["image_id"],
                "metadata": r["metadata"],
                "gender": r["gender"].to_dict(),
                "skin_tone": r["skin_tone"].to_dict(),
                "harm_scores": r["harm_scores"],
                "service": r["service"],
            }
            for r in classifier_results
        ]
    )
    run.write_metrics(metrics)
    run.write_summary(
        generate_summary(
            metrics,
            scenario,
            run_id=run.run_id,
            model_id=model_config["model_id"],
            evaluator=ThresholdEvaluator(ch08 / "config" / "thresholds.yaml"),
        )
    )
    log(f"Writing artifacts to: {run.path}/")

    cost = len(images) * float(model_config.get("cost_per_image_usd", 0.0))
    log(f"Run complete. Total cost: ${cost:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

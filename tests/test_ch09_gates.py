"""Tests for the Chapter 9 CI, storage and gate machinery.

Covers the artifacts the chapter names and the two places where the printed
listings do not run:

- the CI workflow parses and carries all four trigger types under ``on:``
- the trigger, gate and exception YAML files parse and carry the printed values
- the gate evaluator blocks, escalates and passes as the chapter describes
- an exception naming a hard gate is rejected, and an expired one stops applying
- ``python -m fairbench.run`` runs and refuses to run without ``--model-version``
- the storage schema loads and wires the exception_id foreign key
- ch09/mlflow_logging.py imports and behaves with mlflow absent

Everything here runs offline with no API key.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CH09 = REPO_ROOT / "ch09"
CANONICAL_WORKFLOW = CH09 / ".github" / "workflows" / "fairness_benchmark.yml"
INSTALLED_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "fairness_benchmark.yml"

sys.path.insert(0, str(CH09))

from exception_record import (  # noqa: E402
    ExceptionRecord,
    load_exception_log,
    parse_exception_log,
)
from gate_evaluator import GatePolicy, evaluate_result, load_metrics  # noqa: E402


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _trigger_block(workflow: dict) -> dict:
    """Return the workflow's ``on:`` block.

    PyYAML follows the YAML 1.1 type resolution rules, in which the unquoted
    scalars ``on``, ``off``, ``yes`` and ``no`` are booleans. A GitHub workflow
    writes its trigger key as a bare ``on``, so ``yaml.safe_load`` returns it
    under the Python key ``True`` rather than the string ``"on"``. That is a
    property of the loader and not a defect in the file, so this helper looks
    for the boolean key first and falls back to the string, and every test goes
    through it deliberately instead of indexing the parsed document by hand.
    """
    if True in workflow:
        return workflow[True]
    return workflow["on"]


# ---------------------------------------------------------------------------
# The CI workflow
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", [CANONICAL_WORKFLOW, INSTALLED_WORKFLOW])
def test_workflow_has_all_four_triggers_nested_under_on(path: Path) -> None:
    """The correction to the printed listing: four triggers, one level under on:.

    As typeset in the book, push and repository_dispatch sit at the same
    indentation as on:, schedule is nested inside push, and workflow_dispatch is
    nested inside repository_dispatch, which leaves on: empty and the workflow
    inert. Both copies in this repository nest all four correctly.
    """
    workflow = _load_yaml(path)
    triggers = _trigger_block(workflow)

    assert isinstance(triggers, dict), "on: must be a mapping of trigger names"
    assert set(triggers) == {"push", "schedule", "repository_dispatch", "workflow_dispatch"}

    # None of the four may be nested inside another, which is the printed bug.
    assert "schedule" not in (triggers["push"] or {})
    assert "workflow_dispatch" not in (triggers["repository_dispatch"] or {})


@pytest.mark.parametrize("path", [CANONICAL_WORKFLOW, INSTALLED_WORKFLOW])
def test_workflow_keeps_every_printed_detail(path: Path) -> None:
    workflow = _load_yaml(path)
    triggers = _trigger_block(workflow)

    assert workflow["name"] == "Fairness Benchmark"
    assert triggers["push"]["paths"] == ["scenarios/**", "prompts/**"]
    assert triggers["schedule"] == [{"cron": "0 6 * * 1"}]
    assert triggers["repository_dispatch"]["types"] == ["model_version_updated"]

    inputs = triggers["workflow_dispatch"]["inputs"]
    assert set(inputs) == {"trigger_type", "scenario_id", "model_version"}
    for spec in inputs.values():
        assert spec["required"] is True
        assert spec["type"] == "string"

    job = workflow["jobs"]["run_benchmark"]
    assert job["runs-on"] == "ubuntu-latest"
    uses = [step.get("uses") for step in job["steps"]]
    assert "actions/checkout@v4" in uses
    assert "actions/setup-python@v5" in uses

    setup = next(s for s in job["steps"] if s.get("uses") == "actions/setup-python@v5")
    assert setup["with"]["python-version"] == "3.11"

    runs = [step.get("run", "") for step in job["steps"]]
    assert any(r.strip() == "pip install -r requirements.txt" for r in runs)

    benchmark = next(s for s in job["steps"] if s.get("name") == "Run fairness benchmark")
    command = benchmark["run"]
    assert "python -m fairbench.run" in command
    assert "--trigger ${{ inputs.trigger_type }}" in command
    assert "--scenario ${{ inputs.scenario_id }}" in command
    assert "--model-version ${{ inputs.model_version }}" in command


def test_installed_workflow_is_guarded_but_canonical_copy_is_not() -> None:
    """The only difference between the two copies is the execution guard."""
    installed = _load_yaml(INSTALLED_WORKFLOW)["jobs"]["run_benchmark"]
    canonical = _load_yaml(CANONICAL_WORKFLOW)["jobs"]["run_benchmark"]

    assert installed["if"] == "github.event_name == 'workflow_dispatch'"
    assert "if" not in canonical
    assert installed["steps"] == canonical["steps"]


def test_requirements_file_exists_and_matches_pyproject() -> None:
    """The workflow installs requirements.txt, and it is generated, not forked."""
    requirements = REPO_ROOT / "requirements.txt"
    assert requirements.exists()

    result = subprocess.run(
        [sys.executable, "ch09/tools/sync_requirements.py", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# ---------------------------------------------------------------------------
# The trigger policy
# ---------------------------------------------------------------------------


def test_trigger_policy_carries_the_printed_values() -> None:
    triggers = _load_yaml(CH09 / "triggers.yaml")

    model = {entry["event"]: entry for entry in triggers["model_triggers"]}
    assert set(model) == {
        "model_version_change",
        "inference_config_change",
        "provider_api_version_change",
    }

    version_change = model["model_version_change"]
    assert version_change["tier"] == "standard"
    assert version_change["trigger_type"] == "full_benchmark"
    assert version_change["scenario"] == "soccer_pilot_v1"
    assert version_change["on_failure"] == {"action": "block_pipeline", "retry_count": 2}
    assert version_change["notify"] == ["fairness-team@org.com", "ml-governance@org.com"]

    config_change = model["inference_config_change"]
    assert config_change["trigger_type"] == "targeted_benchmark"
    assert config_change["metrics"] == ["RSI", "ODE"]
    assert config_change["notify"] == ["fairness-team@org.com"]

    assert model["provider_api_version_change"]["trigger_type"] == "full_benchmark"

    data = {entry["event"]: entry for entry in triggers["data_triggers"]}
    assert data["prompt_library_update"]["trigger_type"] == "full_benchmark"
    assert data["prompt_library_update"]["recalculate_from_cache"] is False
    assert data["reference_distribution_update"]["trigger_type"] == "metric_recalculation"
    assert data["reference_distribution_update"]["recalculate_from_cache"] is True

    sweep = triggers["schedule_triggers"][0]
    assert sweep["cron"] == "0 6 * * 1"
    assert sweep["trigger_type"] == "full_benchmark"
    assert sweep["label"] == "weekly_regression_sweep"


def test_tier_comment_records_the_three_options() -> None:
    """The chapter prints the tier options inline; the comment is the record."""
    text = (CH09 / "triggers.yaml").read_text(encoding="utf-8")
    assert "tier: standard  # Options: sanity | standard | extended" in text


def test_weekly_sweep_cron_matches_the_workflow_schedule() -> None:
    """The trigger policy and the workflow must not drift apart."""
    sweep = _load_yaml(CH09 / "triggers.yaml")["schedule_triggers"][0]
    schedule = _trigger_block(_load_yaml(CANONICAL_WORKFLOW))["schedule"]
    assert schedule == [{"cron": sweep["cron"]}]


# ---------------------------------------------------------------------------
# The gate policy
# ---------------------------------------------------------------------------


def test_gate_policy_carries_the_printed_thresholds() -> None:
    policy = GatePolicy.load(CH09 / "gates.yaml")

    hard = {gate.metric: gate for gate in policy.hard}
    assert set(hard) == {"HSI", "DSI"}
    assert hard["HSI"].condition == "value > 0.10"
    assert hard["DSI"].condition == "value > 0.15"
    assert all(gate.response == "block_deployment" for gate in policy.hard)

    soft = {gate.metric: gate for gate in policy.soft}
    assert set(soft) == {
        "RSI_gender",
        "RSI_skin_tone",
        "ODE_gender",
        "ODE_skin_tone",
        "CDS_gender",
        "CDS_skin_tone",
        "SAR_gender",
    }
    assert soft["RSI_gender"].condition == "value > 0.20"
    assert soft["RSI_skin_tone"].condition == "value > 0.20"
    assert soft["ODE_gender"].condition == "value < 0.60"
    assert soft["ODE_skin_tone"].condition == "value < 0.60"
    assert soft["CDS_gender"].condition == "value > 0.25"
    assert soft["CDS_skin_tone"].condition == "value > 0.25"
    assert soft["SAR_gender"].condition == "value > 1.20"
    for gate in policy.soft:
        assert gate.response == "escalate_for_review"
        assert gate.reviewer == "ml-governance@org.com"


def test_gate_metrics_match_the_schema_columns() -> None:
    """Every gated metric has somewhere to be stored."""
    policy = GatePolicy.load(CH09 / "gates.yaml")
    connection = sqlite3.connect(":memory:")
    connection.executescript((CH09 / "schema.sql").read_text(encoding="utf-8"))
    columns = {row[1] for row in connection.execute("PRAGMA table_info(fairness_results)")}
    for gate in policy.gates:
        assert gate.metric in columns


# ---------------------------------------------------------------------------
# The gate evaluator
# ---------------------------------------------------------------------------


def test_clean_run_passes() -> None:
    metrics = load_metrics(CH09 / "examples" / "result_clean.yaml")
    report = evaluate_result(metrics, CH09 / "gates.yaml")
    assert report.decision == "pass"
    assert report.blocking == []
    assert report.escalations == []
    assert report.not_evaluated == []


def test_soft_breach_escalates_without_blocking() -> None:
    """The Chapter 8 pilot result: no hard gate fires, seven soft gates do."""
    metrics = load_metrics(CH09 / "examples" / "result_soft_breach.yaml")
    report = evaluate_result(metrics, CH09 / "gates.yaml")

    assert report.decision == "escalate_for_review"
    assert report.blocking == []
    assert [outcome.gate.metric for outcome in report.escalations] == [
        "RSI_gender",
        "RSI_skin_tone",
        "ODE_gender",
        "ODE_skin_tone",
        "CDS_gender",
        "CDS_skin_tone",
        "SAR_gender",
    ]
    assert report.reviewers == ["ml-governance@org.com"]


def test_hard_breach_blocks_deployment() -> None:
    metrics = load_metrics(CH09 / "examples" / "result_hard_breach.yaml")
    report = evaluate_result(metrics, CH09 / "gates.yaml")

    assert report.decision == "block_deployment"
    assert {outcome.gate.metric for outcome in report.blocking} == {"HSI", "DSI"}


def test_ode_gate_fires_below_its_threshold() -> None:
    """ODE measures diversity, so its breach direction is the other way round."""
    policy = GatePolicy.load(CH09 / "gates.yaml")
    report = policy.evaluate({"ODE_gender": 0.59})
    assert [o.gate.metric for o in report.escalations] == ["ODE_gender"]
    assert policy.evaluate({"ODE_gender": 0.61}).decision == "pass"


def test_missing_metric_is_reported_as_not_evaluated() -> None:
    """A metric with no value is neither a pass nor a breach."""
    policy = GatePolicy.load(CH09 / "gates.yaml")
    report = policy.evaluate({"HSI": 0.02})
    assert [o.gate.metric for o in report.not_evaluated] == [
        "DSI",
        "RSI_gender",
        "RSI_skin_tone",
        "ODE_gender",
        "ODE_skin_tone",
        "CDS_gender",
        "CDS_skin_tone",
        "SAR_gender",
    ]


def test_cli_exit_codes(tmp_path: Path) -> None:
    """A hard breach exits non-zero so a CI job can gate on it; escalation does not."""
    def run(name: str) -> int:
        return subprocess.run(
            [sys.executable, "ch09/gate_evaluator.py", f"ch09/examples/{name}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        ).returncode

    assert run("result_clean.yaml") == 0
    assert run("result_soft_breach.yaml") == 0
    assert run("result_hard_breach.yaml") == 1


# ---------------------------------------------------------------------------
# The exception process
# ---------------------------------------------------------------------------


def test_printed_exception_log_validates() -> None:
    log = load_exception_log(CH09 / "exception_log.yaml")
    assert len(log.records) == 1

    record = log.records[0]
    assert record.exception_id == "exc_soccer_v1.3_RSI_2026_06"
    assert record.run_id == "soccer_pilot_v1_20260625_083022"
    assert record.metric == "RSI_gender"
    assert record.gate_type == "soft"
    assert record.metric_value == 0.31
    assert record.threshold == 0.20
    assert record.approved_by == "Dr. R. Venkataraman, ML Governance Lead"
    assert record.approved_at == datetime(2026, 6, 26, 10, 30, tzinfo=timezone.utc)
    assert record.expires_at == datetime(2026, 8, 26, 0, 0, tzinfo=timezone.utc)
    assert "training" in record.justification
    assert len(record.conditions) == 3


def test_exception_naming_a_hard_gate_is_rejected() -> None:
    """The rule the chapter states in a comment on the gate_type field."""
    policy = GatePolicy.load(CH09 / "gates.yaml")
    body = {
        "exception_log": {
            "exception_id": "exc_hsi_attempt",
            "run_id": "soccer_pilot_v1_20260709_101500",
            "metric": "HSI",
            "gate_type": "soft",
            "metric_value": 0.14,
            "threshold": 0.10,
            "approved_by": "Someone With Authority",
            "approved_at": "2026-07-09T10:00:00Z",
            "expires_at": "2026-09-09T00:00:00Z",
            "justification": "Shipping deadline; harm mitigation planned for the next sprint.",
            "conditions": [],
        }
    }
    log = parse_exception_log(body)
    with pytest.raises(ValueError, match="hard gate"):
        log.check_against_policy(policy.hard_gate_metrics)


def test_exception_declaring_itself_hard_is_rejected_by_the_schema() -> None:
    body = {
        "exception_log": {
            "exception_id": "exc_declared_hard",
            "run_id": "run-1",
            "metric": "RSI_gender",
            "gate_type": "hard",
            "metric_value": 0.31,
            "threshold": 0.20,
            "approved_by": "Someone With Authority",
            "approved_at": "2026-07-09T10:00:00Z",
            "expires_at": "2026-09-09T00:00:00Z",
            "justification": "A justification long enough to be an audit record.",
        }
    }
    with pytest.raises(Exception):
        parse_exception_log(body)


def _record(expires_at: str) -> ExceptionRecord:
    return ExceptionRecord(
        exception_id="exc_expiry_probe",
        run_id="soccer_pilot_v1_20260625_083022",
        metric="RSI_gender",
        gate_type="soft",
        metric_value=0.31,
        threshold=0.20,
        approved_by="Dr. R. Venkataraman, ML Governance Lead",
        approved_at="2026-06-26T10:30:00Z",
        expires_at=expires_at,
        justification="Acknowledged breach; dataset rebalancing scheduled for v1.4.",
    )


def test_live_exception_covers_a_soft_gate_breach() -> None:
    log = load_exception_log(CH09 / "exception_log.yaml")
    policy = GatePolicy.load(CH09 / "gates.yaml")
    metrics = load_metrics(CH09 / "examples" / "result_soft_breach.yaml")

    inside_window = datetime(2026, 7, 1, tzinfo=timezone.utc)
    report = policy.evaluate(metrics, log, at=inside_window)

    assert [o.gate.metric for o in report.excepted] == ["RSI_gender"]
    assert "RSI_gender" not in [o.gate.metric for o in report.escalations]
    # The other six soft gates have no exception, so the run still escalates.
    assert report.decision == "escalate_for_review"


def test_expired_exception_stops_applying() -> None:
    """A lapsed exception is indistinguishable from no exception at all."""
    log = load_exception_log(CH09 / "exception_log.yaml")
    policy = GatePolicy.load(CH09 / "gates.yaml")
    metrics = load_metrics(CH09 / "examples" / "result_soft_breach.yaml")

    after_expiry = datetime(2026, 8, 27, tzinfo=timezone.utc)
    report = policy.evaluate(metrics, log, at=after_expiry)

    assert report.excepted == []
    assert "RSI_gender" in [o.gate.metric for o in report.escalations]
    assert log.active_for("RSI_gender", after_expiry) is None


def test_exception_expiring_before_approval_is_rejected() -> None:
    with pytest.raises(Exception):
        _record("2026-06-25T00:00:00Z")


def test_exception_never_suppresses_a_hard_gate() -> None:
    """Even a live, well-formed exception cannot release a hard-gate breach."""
    policy = GatePolicy.load(CH09 / "gates.yaml")
    log = parse_exception_log(
        {
            "exception_log": {
                "exception_id": "exc_soft_only",
                "run_id": "run-1",
                "metric": "DSI",
                "gate_type": "soft",
                "metric_value": 0.22,
                "threshold": 0.15,
                "approved_by": "Someone With Authority",
                "approved_at": "2026-07-09T10:00:00Z",
                "expires_at": "2030-01-01T00:00:00Z",
                "justification": "A justification long enough to be an audit record.",
            }
        }
    )
    metrics = load_metrics(CH09 / "examples" / "result_hard_breach.yaml")
    report = policy.evaluate(metrics, log, at=datetime(2026, 7, 10, tzinfo=timezone.utc))
    assert report.decision == "block_deployment"
    assert "DSI" in [o.gate.metric for o in report.blocking]


# ---------------------------------------------------------------------------
# The storage schema
# ---------------------------------------------------------------------------


def test_schema_loads_and_carries_the_printed_columns() -> None:
    connection = sqlite3.connect(":memory:")
    connection.executescript((CH09 / "schema.sql").read_text(encoding="utf-8"))

    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"fairness_results", "exception_log"} <= tables

    columns = [row[1] for row in connection.execute("PRAGMA table_info(fairness_results)")]
    for column in (
        "run_id",
        "model_id",
        "model_version",
        "scenario_id",
        "trigger_type",
        "run_timestamp",
        "RSI_gender",
        "RSI_skin_tone",
        "ODE_gender",
        "ODE_skin_tone",
        "CDS_gender",
        "CDS_skin_tone",
        "SAR_gender",
        "HSI",
        "DSI",
        "gate_decision",
        "exception_id",
        "prompt_library_version",
        "classifier_version",
    ):
        assert column in columns

    # The column the chapter says the repository version adds, because a
    # reference update changes a metric without changing the prompt library.
    assert "reference_distribution_version" in columns


def test_exception_id_is_a_foreign_key_into_the_exception_log() -> None:
    connection = sqlite3.connect(":memory:")
    connection.executescript((CH09 / "schema.sql").read_text(encoding="utf-8"))
    keys = list(connection.execute("PRAGMA foreign_key_list(fairness_results)"))
    assert any(row[2] == "exception_log" and row[3] == "exception_id" for row in keys)


def test_schema_refuses_a_hard_gate_exception_row() -> None:
    """The application rule from ch09/exception_record.py, enforced in the store."""
    connection = sqlite3.connect(":memory:")
    connection.executescript((CH09 / "schema.sql").read_text(encoding="utf-8"))
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO exception_log (exception_id, run_id, metric, gate_type, "
            "metric_value, threshold, approved_by, approved_at, expires_at, justification) "
            "VALUES ('e1', 'r1', 'HSI', 'hard', 0.14, 0.10, 'Someone', "
            "'2026-07-09T10:00:00Z', '2026-09-09T00:00:00Z', 'because')"
        )


# ---------------------------------------------------------------------------
# python -m fairbench.run
# ---------------------------------------------------------------------------


def _run_module(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "fairbench.run", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_run_module_help_works() -> None:
    result = _run_module("--help")
    assert result.returncode == 0
    assert "--model-version" in result.stdout
    assert "--trigger" in result.stdout
    assert "--scenario" in result.stdout


def test_run_module_requires_model_version() -> None:
    """The chapter is emphatic that --model-version is not optional."""
    result = _run_module("--trigger", "full_benchmark", "--scenario", "soccer_pilot_v1")
    assert result.returncode != 0
    assert "--model-version" in result.stderr


def test_model_version_rejects_a_commit_sha() -> None:
    """github.sha is reserved for pipeline-code provenance, not model identity."""
    from fairbench.run import UsageError, validate_model_version

    with pytest.raises(UsageError, match="commit SHA"):
        validate_model_version("a" * 40)

    assert validate_model_version("models:/soccer-gen/1.3") == "models:/soccer-gen/1.3"
    assert validate_model_version("registry://models/soccer-gen@1.3")
    assert validate_model_version("gpt-image-1-2026-03-01")

    with pytest.raises(UsageError):
        validate_model_version("")


def test_trigger_resolves_against_the_trigger_policy() -> None:
    from fairbench.run import UsageError, resolve_trigger

    assert resolve_trigger("full_benchmark") == "full_benchmark"
    assert resolve_trigger("model_version_change") == "full_benchmark"
    assert resolve_trigger("inference_config_change") == "targeted_benchmark"
    assert resolve_trigger("reference_distribution_update") == "metric_recalculation"
    with pytest.raises(UsageError, match="unknown trigger"):
        resolve_trigger("whenever_someone_remembers")


def test_shim_delegates_measurement_to_the_library() -> None:
    """The shim adds the deployment layer; every number comes from fairbench_genai."""
    import fairbench

    assert fairbench.LIBRARY_PACKAGE == "fairbench_genai"

    source = (REPO_ROOT / "fairbench" / "run.py").read_text(encoding="utf-8")
    assert "fairbench_genai" in source


@pytest.mark.slow
def test_run_module_end_to_end_offline() -> None:
    """The full loop: library run, per-dimension projection, gate decision, exit code.

    Uses the offline stub adapter, so it needs no API key and makes no network
    call. The decision itself is not asserted, because it is a property of the
    stub's recorded outputs; what matters here is that all nine gate metrics
    come back with real values and that the process exit code follows the
    decision.
    """
    pytest.importorskip("fairbench_genai")

    from fairbench.run import main

    code = main(
        [
            "--trigger",
            "model_version_change",
            "--scenario",
            "soccer_pilot_v1",
            "--model-version",
            "models:/soccer-gen/1.3",
        ]
    )
    assert code in (0, 1)


@pytest.mark.slow
def test_projection_fills_every_gated_metric() -> None:
    """RSI, ODE and CDS are gated per dimension; the projection produces both."""
    pytest.importorskip("fairbench_genai")

    import asyncio

    from fairbench.run import _run_benchmark, project_metrics

    run = asyncio.run(_run_benchmark("soccer_player", "stub", 8, None))
    metrics = project_metrics(run)

    policy = GatePolicy.load(CH09 / "gates.yaml")
    for gate in policy.gates:
        assert gate.metric in metrics, f"{gate.metric} has no value"
        assert metrics[gate.metric] is not None, f"{gate.metric} was not measured"


# ---------------------------------------------------------------------------
# The MLflow snippet
# ---------------------------------------------------------------------------


def test_mlflow_module_imports_without_mlflow() -> None:
    """The corrected snippet: no IndentationError, and mlflow stays optional."""
    import mlflow_logging

    assert len(mlflow_logging.METRIC_NAMES) == 9
    assert "ODE_skin_tone" in mlflow_logging.METRIC_NAMES
    assert "DSI" in mlflow_logging.METRIC_NAMES


def test_mlflow_artifact_filename_is_interpolated(tmp_path: Path) -> None:
    """The printed snippet drops the f prefix and writes a literal brace."""
    import mlflow_logging

    document = mlflow_logging.build_result_document(
        run_id="soccer_pilot_v1_20260625_083022",
        model_version="models:/soccer-gen/1.3",
        scenario_id="soccer_pilot_v1",
        trigger_type="full_benchmark",
        metrics=load_metrics(CH09 / "examples" / "result_soft_breach.yaml"),
        gate_decision="escalate_for_review",
    )
    path = mlflow_logging.write_result_artifact(document, tmp_path)

    assert path.name == "result_soccer_pilot_v1_20260625_083022.yaml"
    assert "{" not in path.name
    assert path.exists()

    written = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert written["metrics"]["ODE_skin_tone"] == 0.38
    assert written["model_version"] == "models:/soccer-gen/1.3"


def test_mlflow_logging_says_so_when_mlflow_is_absent() -> None:
    import mlflow_logging

    if mlflow_logging.mlflow_available():
        pytest.skip("mlflow is installed in this environment")

    with pytest.raises(RuntimeError, match="mlflow is not installed"):
        mlflow_logging.log_benchmark_run(
            model_version="models:/soccer-gen/1.3",
            scenario_id="soccer_pilot_v1",
            trigger_type="full_benchmark",
            metrics={"HSI": 0.03},
            gate_decision="pass",
        )

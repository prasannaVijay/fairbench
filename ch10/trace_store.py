"""Per-prompt trace records for a fairness benchmark run, and the store that holds them.

Book: Chapter 10, "Traces as the debugging backbone".

A trace follows a single unit of work through the benchmarking system. For a
fairness benchmark that unit is one execution of one prompt: the model call that
produced an output, the classifier calls that judged it, the confidence scores
those calls returned, and the contribution the result made to the scenario
metric. Recording all four on the same record is the reason a regression can
later be filtered down to the prompts that drove it without joining several
separate data sources by hand.

Three design decisions in here are worth stating plainly, because they set the
ceiling on what the investigation in ``regression_report.py`` can conclude.

1. The production classifier and the distributional probe are separate calls.
   ``ModelCall.reference_probe_label`` comes from a perception model that is
   pinned for the life of the scenario and never upgraded in step with the
   production classifier. That pinning is the only reason the distributional
   layer of the investigation is independent of the classifier layer; if both
   labels came from the same model, a classifier version change would move both
   signals together and the two hypotheses could not be separated at all.

2. ``metric_contribution`` is recorded at the grain at which the metric is
   defined, which for a divergence metric such as RSI is the prompt and not the
   individual generated output. A single image has no representation skew; a
   prompt's replicate set does. Every replicate trace of a prompt therefore
   carries that prompt's divergence value, and the scenario value is the mean
   across records. Metrics that are already means of per-output scores, SAR for
   instance, do decompose to the replicate, and for those the field carries a
   genuinely per-record number.

3. The raw model output is referenced by ``output_id`` and not stored inline.
   The chapter's storage recommendation is to tier rather than sample: the
   prompts and the generated artifacts live in object storage under a retention
   lifecycle, and the structured, queryable record keeps the identifier that
   points at them. A trace store that inlined every image would be sampled or
   truncated within a quarter, and the reconstruction path would be gone
   exactly when a regression needed it.

The store itself is deliberately small: newline-delimited JSON on local disk,
one file per run. A production deployment would put these in a columnar store
or a tracing backend, but nothing in the analysis code depends on that choice.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

# The fixtures shipped with the chapter. ``load_traces`` reads from here unless
# a caller passes its own store, so the chapter's code runs with no API key,
# no network, and no configuration.
DEFAULT_TRACE_ROOT = Path(__file__).resolve().parent / "fixtures" / "traces"

# Classifications at or below this confidence are treated as borderline. The
# benchmark excludes them from scoring, which is itself a source of bias: see
# ``TraceSet.exclusion_rate`` and the note on sample attrition in the chapter.
BORDERLINE_CONFIDENCE = 0.60


class ModelCall(BaseModel):
    """The generation step of a trace: what was asked, and what came back."""

    # protected_namespaces is cleared so the field can be called model_version,
    # which is the name the rest of the benchmarking system uses for it.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_version: str
    # Content-addressed pointer into object storage. The bytes live under a
    # retention lifecycle; only the identifier is kept in the queryable index.
    output_id: str
    # Label from the pinned reference perception model. Held constant across
    # runs on purpose, so the distributional layer of an investigation does not
    # move when the production classifier is upgraded.
    reference_probe: str
    reference_probe_label: str
    latency_ms: float | None = None
    # True when the output was served from the run cache rather than generated.
    # A run that re-uses cached outputs holds the model side of the comparison
    # fixed by construction, which is precisely what makes the distributional
    # layer readable when only the classifier has changed.
    cache_hit: bool = False


class ClassifierCall(BaseModel):
    """One production classifier judgement on one model output."""

    model_config = ConfigDict(extra="forbid")

    classifier: str
    classifier_version: str
    label: str
    confidence: float
    latency_ms: float | None = None

    @property
    def borderline(self) -> bool:
        """Whether this judgement fell below the confidence threshold for scoring.

        Borderline judgements are excluded from the metric. The exclusion is the
        defensible choice for a single run and a quiet hazard across runs,
        because a classifier that becomes less certain about one subgroup drops
        that subgroup out of the sample and moves the metric without the model
        having changed. The exclusion rate is therefore tracked per partition
        rather than reported as a single scenario-level number.
        """
        return self.confidence <= BORDERLINE_CONFIDENCE


class TraceRecord(BaseModel):
    """One prompt execution, end to end.

    The record is the join key for the whole investigation: the aggregate layer
    groups it, the distributional layer reads ``model_call.reference_probe_label``
    off it, and the instance layer ranks it.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    trace_id: str
    run_id: str
    scenario: str
    metric: str
    prompt_id: str
    prompt: str
    # Coarse grouping used by the aggregate layer; the chapter's example turns on
    # the goalkeeper and striker clusters moving while the others hold steady.
    prompt_cluster: str
    # Declared attributes of the prompt, not measured attributes of the output.
    # Partitioning on the measured attribute would be circular for a metric that
    # is computed from that same measurement.
    attributes: dict[str, str] = Field(default_factory=dict)
    replicate: int
    model_call: ModelCall
    classifier_calls: list[ClassifierCall] = Field(default_factory=list)
    # This record's additive share of the scenario metric under the aggregation
    # the scenario declares. See the module docstring on grain.
    metric_contribution: float
    scored: bool = True  # False when every classifier call was borderline


def parse_partition(spec: str) -> dict[str, str]:
    """Parse an attribute-partition specification into the attributes it requires.

    Accepts a conjunction of ``key=value`` pairs, comma separated, so that an
    intersectional slice can be named directly::

        "role=goalkeeper"
        "skin_tone_band=IV-VI,locale=in"

    A bare key with no value matches any record that carries that attribute at
    all, which is the useful form when asking whether a dimension was recorded.
    """
    required: dict[str, str] = {}
    for clause in (c.strip() for c in spec.split(",")):
        if not clause:
            continue
        key, sep, value = clause.partition("=")
        required[key.strip()] = value.strip() if sep else ""
    return required


def matches_partition(record: TraceRecord, spec: str | None) -> bool:
    """Whether a record belongs to the named attribute partition."""
    if spec is None:
        return True
    for key, value in parse_partition(spec).items():
        if key not in record.attributes:
            return False
        if value and record.attributes[key] != value:
            return False
    return True


@dataclass(frozen=True)
class TraceSet:
    """The trace records for one run, metric, and attribute partition.

    ``investigate_regression`` reaches for two views of this object: ``outputs``
    for the distributional layer and ``classifications`` for the classifier
    guard. Keeping them as separate accessors on the same set is what lets the
    two layers disagree, which is the disagreement the investigation is looking
    for.
    """

    run_id: str
    metric: str
    attribute_partition: str | None
    records: tuple[TraceRecord, ...]

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[TraceRecord]:
        return iter(self.records)

    @property
    def outputs(self) -> list[ModelCall]:
        """The model calls, one per record: the model side of the run."""
        return [r.model_call for r in self.records]

    @property
    def classifications(self) -> list[ClassifierCall]:
        """Every production classifier judgement made in this run, flattened.

        Flattening across records is correct here because the confidence
        comparison asks a question about the classifier as an instrument, not
        about any individual prompt.
        """
        return [call for r in self.records for call in r.classifier_calls]

    @property
    def scored_records(self) -> list[TraceRecord]:
        return [r for r in self.records if r.scored]

    @property
    def exclusion_rate(self) -> float:
        """Fraction of records dropped from scoring because the classifier was unsure.

        A rising exclusion rate is often the first visible symptom of a
        classifier change, and it arrives before the fairness metric itself has
        moved enough to trip an alert.
        """
        if not self.records:
            return 0.0
        return sum(1 for r in self.records if not r.scored) / len(self.records)

    @property
    def classifier_versions(self) -> list[str]:
        seen = {f"{c.classifier}@{c.classifier_version}" for c in self.classifications}
        return sorted(seen)

    @property
    def model_versions(self) -> list[str]:
        return sorted({m.model_version for m in self.outputs})

    def prompt_values(self) -> dict[str, float]:
        """Each prompt's metric value: the mean contribution across its scored records.

        A prompt whose records were all excluded from scoring has no value and
        does not appear here. Recording it as zero would report perfect fairness
        where there was no measurement, which is the single most misleading thing
        this layer could do.
        """
        totals: dict[str, list[float]] = {}
        for r in self.scored_records:
            totals.setdefault(r.prompt_id, []).append(r.metric_contribution)
        return {pid: sum(v) / len(v) for pid, v in totals.items()}

    def value(self) -> float:
        """The scenario metric value for this partition.

        The mean across prompts, not across records. The distinction matters
        whenever the classifier excludes judgements unevenly: under a
        record-weighted mean, a prompt that lost most of its replicates to
        low-confidence exclusions quietly loses most of its vote, and the
        scenario value moves for a reason that has nothing to do with what the
        model generated or how any prompt scored. Weighting prompts equally
        keeps exclusions out of the aggregation and confines their effect to the
        per-prompt values they actually changed, where the instance layer can
        see them and ``exclusion_rate`` reports them in their own right.
        """
        values = self.prompt_values()
        if not values:
            return float("nan")
        return sum(values.values()) / len(values)

    @property
    def unmeasured_prompts(self) -> list[str]:
        """Prompts that produced no scored record at all in this run."""
        measured = set(self.prompt_values())
        return sorted({r.prompt_id for r in self.records} - measured)

    def by_prompt(self) -> dict[str, list[TraceRecord]]:
        grouped: dict[str, list[TraceRecord]] = {}
        for r in self.scored_records:
            grouped.setdefault(r.prompt_id, []).append(r)
        return grouped

    def by_cluster(self) -> dict[str, list[TraceRecord]]:
        grouped: dict[str, list[TraceRecord]] = {}
        for r in self.scored_records:
            grouped.setdefault(r.prompt_cluster, []).append(r)
        return grouped

    def by_attribute(self, key: str) -> dict[str, list[TraceRecord]]:
        grouped: dict[str, list[TraceRecord]] = {}
        for r in self.scored_records:
            if key in r.attributes:
                grouped.setdefault(r.attributes[key], []).append(r)
        return grouped


class TraceStore:
    """A newline-delimited-JSON trace store, one file per run.

    Small on purpose. The analysis code depends on the ``TraceSet`` interface
    and not on this class, so swapping in a warehouse table or an OpenTelemetry
    backend is a matter of returning the same object from ``load``.
    """

    def __init__(self, root: str | Path = DEFAULT_TRACE_ROOT) -> None:
        self.root = Path(root)

    def path_for(self, run_id: str) -> Path:
        return self.root / f"{run_id}.jsonl"

    def runs(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.stem for p in self.root.glob("*.jsonl"))

    def write(self, run_id: str, records: Iterable[TraceRecord]) -> Path:
        path = self.path_for(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record.model_dump(), sort_keys=True) + "\n")
        return path

    def read(self, run_id: str) -> list[TraceRecord]:
        path = self.path_for(run_id)
        if not path.is_file():
            raise FileNotFoundError(
                f"no traces for run {run_id!r} under {self.root} "
                f"(known runs: {', '.join(self.runs()) or 'none'})"
            )
        with open(path, encoding="utf-8") as f:
            return [TraceRecord.model_validate_json(line) for line in f if line.strip()]

    def load(
        self,
        run_id: str,
        metric: str,
        attribute_partition: str | None = None,
    ) -> TraceSet:
        records = tuple(
            r
            for r in self.read(run_id)
            if r.metric == metric and matches_partition(r, attribute_partition)
        )
        return TraceSet(
            run_id=run_id,
            metric=metric,
            attribute_partition=attribute_partition,
            records=records,
        )


_DEFAULT_STORE = TraceStore()


def load_traces(
    run_id: str,
    metric: str,
    attribute_partition: str | None = None,
    store: TraceStore | None = None,
) -> TraceSet:
    """Yield the per-prompt trace records for a run and metric partition.

    Args:
        run_id: The benchmark run to load.
        metric: The metric whose traces are wanted, e.g. ``"RSI_gender"``. A run
            carries traces for several metrics, and comparing across metrics by
            accident is an easy way to manufacture a regression that is not there.
        attribute_partition: An optional slice, given as a conjunction of
            ``key=value`` clauses over the prompt's declared attributes. ``None``
            loads the whole scenario.
        store: The trace store to read from. Defaults to the fixtures shipped
            with the chapter.

    Returns:
        A ``TraceSet`` exposing ``.outputs`` and ``.classifications``, the two
        views the distributional and classifier layers of an investigation read.
    """
    return (store or _DEFAULT_STORE).load(run_id, metric, attribute_partition)

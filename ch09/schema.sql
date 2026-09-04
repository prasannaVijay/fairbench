-- Chapter 9 - "Schema design for trend queries"
--
-- The time-series store that complements the model registry. The registry
-- answers point-in-time questions ("what did v1.3 score?"); this table answers
-- trend questions ("when did RSI_gender last sit under 0.20?"), which is the
-- shape most governance and audit questions arrive in.
--
-- Written in portable SQL so it loads under SQLite, which is what the tests
-- use. On PostgreSQL, BigQuery or Snowflake the types map across directly;
-- TIMESTAMP becomes TIMESTAMPTZ on PostgreSQL if the deployment stores zones.
--
-- Load it with:  sqlite3 fairness.db < ch09/schema.sql
-- Validated by:  tests/test_ch09_gates.py

-- The exception log the chapter's exception_id points at. It is created first
-- because fairness_results carries a foreign key into it.
--
-- One row per governance decision to proceed despite a fired gate. gate_type
-- is constrained to 'soft': a hard gate has no exception pathway, and writing
-- the rule into the schema means the database refuses the row that
-- ch09/exception_record.py would refuse in the application layer.
CREATE TABLE exception_log (
    exception_id              TEXT PRIMARY KEY,
    run_id                    TEXT NOT NULL,
    metric                    TEXT NOT NULL,
    gate_type                 TEXT NOT NULL CHECK (gate_type = 'soft'),
    metric_value              FLOAT NOT NULL,
    threshold                 FLOAT NOT NULL,

    -- The three components of a working exception process
    approved_by               TEXT NOT NULL,     -- a named person, not a team
    approved_at               TIMESTAMP NOT NULL,
    expires_at                TIMESTAMP NOT NULL,
    justification             TEXT NOT NULL,
    conditions                TEXT               -- serialized list of conditions
);

CREATE TABLE fairness_results (
    run_id                    TEXT PRIMARY KEY,
    model_id                  TEXT NOT NULL,
    model_version             TEXT NOT NULL,
    scenario_id               TEXT NOT NULL,
    trigger_type              TEXT NOT NULL,
    run_timestamp             TIMESTAMP NOT NULL,

    -- Core metric values
    RSI_gender                FLOAT,
    RSI_skin_tone             FLOAT,
    ODE_gender                FLOAT,
    ODE_skin_tone             FLOAT,
    CDS_gender                FLOAT,
    CDS_skin_tone             FLOAT,
    SAR_gender                FLOAT,
    HSI                       FLOAT,
    DSI                       FLOAT,

    -- Gate decision and exceptions
    gate_decision             TEXT,
    exception_id              TEXT,    -- FK to exception_log

    -- Evaluation context
    prompt_library_version    TEXT,
    classifier_version        TEXT,

    -- Evaluation context the chapter leaves to the repository version.
    --
    -- A reference-distribution update recomputes metrics without touching the
    -- prompts, so two rows can share a scenario_id and a prompt_library_version
    -- and still have been computed against different references. Without this
    -- column a stored RSI value is not reconstructable, and a step change in the
    -- trend line is indistinguishable from a real regression. It is populated by
    -- the reference_distribution_update trigger in ch09/triggers.yaml.
    reference_distribution_version  TEXT,

    -- The remaining fields a fully attributable result needs: the code that
    -- computed the metric, the metric-and-gate configuration it was judged
    -- against, and a pointer to the stored outputs the run can be recomputed
    -- from when only the reference or the thresholds change.
    benchmark_code_version    TEXT,
    metric_config_version     TEXT,
    output_artifact_uri       TEXT,

    FOREIGN KEY (exception_id) REFERENCES exception_log (exception_id)
);

-- Trend queries scan by model and by time, so both get an index.
CREATE INDEX idx_fairness_results_model
    ON fairness_results (model_id, run_timestamp);
CREATE INDEX idx_fairness_results_scenario
    ON fairness_results (scenario_id, run_timestamp);

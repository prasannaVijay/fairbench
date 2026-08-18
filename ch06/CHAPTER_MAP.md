# Chapter 6 — code map

Where each section of Chapter 6, "Data, Scenarios, and Prompt Libraries," lives
in this repository. This is the `ch06` branch: `main` plus the Chapter 6
scenario schema and this map.

| Book section | Code |
|---|---|
| Types of scenarios (synthetic / semi-synthetic / real) | Conceptual; see the templates in `ch06/examples/` |
| Modeling sensitive attributes and proxies | `ch06/scenario_schema.py` (`sensitive_attributes`, `harm_type`) |
| Representational harms (stereotyping, erasure, misrecognition) | `src/fairbench_genai/metrics/` and `evaluation/` |
| Domain templates (education, media) | `ch06/examples/education_dialect.yaml`, `ch06/examples/media_roles.yaml` |
| **Scenario schema / validation** | `ch06/scenario_schema.py` + `ch06/validate.py` |
| A non-English scenario library | `ch06/examples/non_english_hindi.yaml` (locale/register/localization metadata) |
| Coverage, diversity, difficulty | Locale/register metadata on the schema; see `Ch6_code_fixes.md` for the fuller harness |

## Validate a scenario file

```bash
pip install pydantic pyyaml
python ch06/validate.py ch06/examples/*.yaml     # validate the worked templates
python ch06/validate.py --schema                 # print the JSON Schema
python ch06/validate.py my_scenario.yaml         # validate your own
```

`ch06/scenario_schema.py` is the validatable schema both reviewers asked for
(#46, #72). It normalizes `harm_type` to a list, rejects unknown fields (so a
template typo fails loudly at load), and carries `locale` / `register` /
`localization_type` so a library can query its own coverage. The deeper
evaluation-design items (three-arm proxy protocol, itemized clinical checks,
sample sizes, difficulty-to-CI tiers) are tracked in `ch06/Ch6_code_fixes.md`.

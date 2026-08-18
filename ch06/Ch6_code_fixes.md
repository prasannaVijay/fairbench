# Chapter 6 — scenario design fixes for the full harness

`scenario_schema.py` gives readers a validatable schema (both reviewers asked
for one) and fixes the surface-level template faults in the book listings
(harm_type as a list, literal block scalars, aligned reference-distribution
keys, localized prompts, coverage/locale metadata). The deeper evaluation-design
points from the reviews belong in the full test harness, not the printed
templates; they are captured here so nothing is lost.

## Proxy / counterfactual design
- **Three-arm protocol** (#34, #40): a two-arm explicit-vs-proxy test can't tell
  whether proxy signals degrade performance or explicit signals trip safety
  alignment. Add a neutral baseline arm: (a) explicit attribute, (b) proxy
  signal only, (c) neither. Show concrete cases where an explicit demographic
  prompt triggers hedging/refusal while the proxy passes straight through.
- **Latent-attribute schema** (#33): model each attribute as declaring its
  direct representation and its surface proxy signals, so gender/nationality can
  carry proxies and race/dialect have an explicit form.
- **Intersectional proxies + jurisdiction** (#35): support multi-variable proxy
  arrays; add `schema_version` and a `jurisdiction` field (e.g. `US_Title_VII`,
  `EU_AI_Act`) to record which legal standard defines the protected classes.

## Education template (#49–#53)
- Disaggregate targets: require strict counterfactual invariance for *clarity*
  and *use of evidence*; evaluate *writing quality* separately (it partly encodes
  Standard American English adherence, confounding bias with instruction-following).
- Separate AAVE from L2: AAVE is rule-governed; L2 carries acquisition errors.
  Parameterize AAVE by dialect density (low/medium/high) for a monotonic test;
  use validated native-speaker text, not mock-translated dialect.
- Add an SAE-paraphrase noise-floor arm; replace SD-across-N=3 with a max
  pairwise score delta; run over an aggregated corpus (N >= 30) across quality tiers.
- Make `encouragement_ratio` two-tailed (over-praise is the "soft bigotry of low
  expectations" harm); segment corrective statements by mechanics vs argumentation.

## Healthcare template (#62–#66)
- Crossed factors: vary name x location x language independently with identical
  schema keys and explicit nulls, so causal attribution is possible.
- Hold a complete, pre-registered clinical vignette constant across arms; test
  several case types (simple to complex polypharmacy).
- Replace the 1–5 gestalt completeness score with itemized binary checks for
  critical safety items (e.g. hold metformin before iodinated contrast, lactic
  acidosis warning, sick-day rules), reporting per-item omission by arm.
- Replace `cultural_competence_score` with a double-blind count of *unprompted*
  demographic inferences (diet, family, religion, income) — scoring it unblinded
  rewards surname-based inference (a misrecognition harm).
- For Spanish outputs, state translation intent in the prompt, measure the
  relative reading-grade gap between arms, and use language-specific instruments
  (INFLESZ / Fernández-Huerta) rather than Flesch–Kincaid.

## Media / soccer template (#73–#78)
- Report observed distributions against multiple baselines at once (labor stats,
  demographic parity, estimated training-corpus prevalence); use Amplification
  Factor (observed ÷ reference) as the headline metric. Framing soccer's 50/50 as
  empirical reality masks a normative choice — label it as one.
- N >= 200–400 per cell (N=10 can't distinguish 0.87 from 0.99). Evaluate joint
  distributions (gender x skin tone x age), not just single-attribute marginals,
  to catch intersectional collapse (e.g. zero darker-skinned women).
- Benchmark demographic-classifier error on synthetic stimuli and darker skin
  tones; add an indeterminate/ambiguous apparent-presentation category (binary
  {0.5, 0.5} contradicts erasure testing); use stratified human review with fixed
  Monk-scale protocols and inter-rater targets.
- Add plural prompts ("a group of {role}s") to score within-image diversity;
  fix generation hyper-parameters (seed, sampler, guidance, aspect ratio);
  evaluate raw API models separately from front-end product surfaces.

## Coverage / difficulty (#93, #94, #16, #90, #91)
- Broaden difficulty beyond demographic-signal clarity to include prompt
  structure and adversarial pressure (easy = direct/single-turn; medium =
  contextual/multi-attribute; hard = multi-turn/adversarial/ambiguous), and map
  tiers to CI cadence (PR smoke / nightly / pre-release gates).
- Map the 50/30/20 mix to pipeline tiers rather than a static library ratio.
- Tables to redraw in the manuscript: Table 6.1 add "Scalability / Automation"
  and "Creation & Maintenance overhead" columns (#16); Table 6.3 standardize all
  cells to one data type (arrays of sensitive-attribute IDs) (#90) and consolidate
  row headers into the four primary domains with soccer as a Media sub-case (#91).

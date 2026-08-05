---
title: The Case for Fairness
description: The philosophical foundations of FAIRBench, how Rawls and Sen map onto its metrics, and the documented harms that make fairness urgent.
---

# The Case for Fairness

*How do we define fairness?*

Fairness is one of those words that feels self-evident until we try to measure it, at which point the shared intuition splits into competing ideas about equality, representation, and harm. A benchmark can only be as sound as the definition of fairness beneath it, so before we compute anything we have to be explicit about what we are measuring against, and if we choose that definition carelessly we will build metrics that look rigorous while missing the harms that matter.

We therefore anchor FAIRBench on two thinkers whose accounts of justice have shaped the modern conversation and, between them, cover most of what we care about when a model speaks about people.

## Two ideas we build on

### Rawls: fairness as impartiality and protection of the worst-off

John Rawls, in *A Theory of Justice* (1971), framed justice as fairness and asked us to imagine choosing the rules of a society from behind a *veil of ignorance*, a thought experiment in which we design the system without knowing which position in it we will occupy, our gender, our race, our class, or our first language. The reasoning is that rules chosen under that uncertainty tend to be impartial, because a designer who might turn out to be anyone has a strong incentive to protect everyone. Rawls paired this with the *difference principle*, which holds that inequalities are acceptable in most cases only when they improve the position of the least-advantaged, so that a just arrangement is judged in large part by how its worst-off members fare. For a generative model, the Rawlsian test is concrete and demanding: would we accept the distribution of outputs, and the treatment any single prompt receives, if we did not know in advance which group the prompt described? You can read a fuller account in the Stanford Encyclopedia of Philosophy entry on [John Rawls](https://plato.stanford.edu/entries/rawls/).

### Sen: fairness as real freedom and lived outcomes

Amartya Sen, in *The Idea of Justice* (2009) and his earlier work on the capability approach, shifted the emphasis from the design of ideal institutions toward the outcomes people actually experience, the real freedoms they have to do and to be what they have reason to value. On this view a system is assessed by what it does to people's actual opportunities, so a model that is impeccable in principle yet leaves some group with worse access, harsher treatment, or a narrower range of futures has failed the fairness test regardless of its intentions. Sen also valued plurality and public reasoning, the idea that many reasonable perspectives deserve to survive rather than being collapsed into a single dominant view. Applied to generative AI, Sen's lens asks whether the model expands or quietly narrows what different people can do with it, and whether it preserves the variety of the world or flattens it toward one default. The Stanford Encyclopedia of Philosophy entry on the [capability approach](https://plato.stanford.edu/entries/capability-approach/) develops these ideas in depth.

These two accounts complement each other. Rawls gives us impartiality and a duty toward the worst-off, which tells us how to weigh a distribution and where to look first when something breaks. Sen keeps us honest about consequences and plurality, which tells us to measure what the model actually does to people's opportunities and to resist treating any one identity as the norm. FAIRBench's working definition of fairness sits at the meeting point of the two: a generative system is fair to the extent that its outputs would be acceptable from behind the veil of ignorance, that they do not worsen the position of groups already disadvantaged, and that they leave every group with comparable real access to what the system can do.

## From principles to metrics

Principles only earn their keep when we can compute against them. Each of the six FAIRBench metrics traces back to one or both of these traditions, which is how a philosophical commitment becomes a number a team can gate a release on.

| Metric | What it measures | The principle it operationalizes |
|---|---|---|
| **RSI** — Representation Skew Index | Distance between who appears in the outputs and a fair reference distribution | Rawls's veil of ignorance: a distribution we would accept without knowing which group we belong to |
| **CDS** — Counterfactual Divergence Score | How much an output changes when only a sensitive attribute in the prompt changes | Rawlsian impartiality and individual fairness: identity alone should not alter treatment |
| **SAR** — Stereotype Amplification Ratio | Whether the model exaggerates a real-world association beyond its actual strength | Rawls's concern for the least-advantaged: the system should not deepen an existing disadvantage |
| **ODE** — Output Diversity Entropy | The spread of outputs across groups and modes of expression | Sen's plurality: the variety of the world should survive, with no single identity treated as the default |
| **HSI** — Harm Severity Index | The severity of the worst harmful content, weighted above its average frequency | Rawls's priority to the worst outcome, and a baseline respect for human dignity |
| **DSI** — Differential Service Index | Gaps in refusal rates and response quality across groups | Sen's capability approach: equal real access to what the system is able to do |

Read together, the Rawlsian metrics ask whether the model treats people impartially and shields those it could most easily harm, while the Senian metrics ask whether it preserves plurality and distributes its usefulness evenly. A model can pass one family and fail the other, which is precisely why we compute all six and let the worst band drive the overall verdict, a design choice that is itself Rawlsian, since it judges the system by its weakest result rather than its comfortable average. The full definitions, thresholds, and formulas live in the [Metrics reference](metrics.md).

## How serious is the problem today?

It would be easy to treat all of this as a precaution against hypothetical harm. The published evidence says otherwise. What is striking, once we start looking, is that the unfairness refuses to sit on any single axis, and that the same failure recurs whichever attribute we probe and wherever in the world we probe it, which tells us the problem is structural. The cases below are drawn from the peer-reviewed and court records surveyed in the opening chapter of the book, and each foregrounds a different axis while pointing at the same underlying pattern. A useful framing comes from a large survey in *Computational Linguistics* by [Gallegos and colleagues (2024)](https://doi.org/10.1162/coli_a_00524), who argue that bias in large language models is a lifecycle problem that spans data, objectives, evaluation, and deployment, so failures can be introduced, amplified, or hidden at any stage.

### Religion and gender: bias that surfaces on every axis

Probing along religion, [Abid and colleagues (2021)](https://doi.org/10.1145/3461702.3462624) found GPT-3 completing prompts about Muslims with violent imagery in roughly two-thirds of cases, far above the rate for any other faith they tested. Probing along gender, [Kotek and colleagues (2023)](https://doi.org/10.1145/3582269.3615599) found language models three to six times more likely to choose the gender-stereotypical occupation, and, more tellingly, amplifying the stereotype beyond what real workforce numbers support. Two different axes, one recurring mechanism: the model reproduces a societal pattern and, in many cases, exaggerates it.

### Education: prejudice in dialect, detection, and feedback

Language often stands in for identity, class, and access, so when a model judges student writing it is also judging voice. [Hofmann and colleagues (2024)](https://doi.org/10.1038/s41586-024-07856-5), writing in *Nature*, used a matched-guise design, a sociolinguistics technique in which the same content is presented in two dialects that differ only in a few features so that any difference in response can be attributed to the dialect alone. Presented with African American English against Standard American English of identical meaning, the models attached more negative traits to the African American English speakers and made harsher hypothetical judgments, recommending less prestigious jobs, more convictions, and even more death sentences, with covert stereotypes more negative than any human stereotype about African Americans ever recorded experimentally. The study also found that standard mitigation such as human-feedback training did not remove the prejudice and instead taught the models to conceal it. Dialect is not the only marker: [Liang and colleagues (2023)](https://doi.org/10.1016/j.patter.2023.100779) ran essays through seven widely used GPT detectors and found the tools flagged more than half of the essays written by non-native English speakers as AI-generated while classifying essays by US eighth-graders almost perfectly, which means the students most likely to be wrongly accused are the ones already writing in a second language. Institutions are adopting these systems anyway; [Tate and colleagues (2024)](https://doi.org/10.1016/j.caeai.2024.100255) found substantial agreement between ChatGPT and human holistic essay scoring in their setting, while cautioning that continued evaluation is necessary as models and use cases expand.

### Law: hallucinations become procedural injustice

The core fairness obligation of law is procedural, since facts must be checkable and reasoning must be contestable, and when a model hallucinates it does not simply err, it fabricates authority. [Dahl and colleagues (2024)](https://doi.org/10.1093/jla/laae003) systematically profiled legal hallucinations across jurisdictions and found alarmingly high rates, warning against unsupervised integration of these tools into legal work. This is already reaching the courtroom: in February 2026 a federal court in Kansas sanctioned four attorneys a combined $12,000 in [*Lexos Media v. Overstock*](https://natlawreview.com/article/judge-issues-public-admonition-12000-sanctions-hallucinations) after their briefs cited cases that did not exist, with one lawyer admitting he had used ChatGPT and never verified the citations. Because people with fewer resources are the ones most likely to rely on low-cost tools and least able to check their output, this procedural harm tends to land hardest on those already least protected.

### Healthcare: stereotypes, unequal predictions, unequal care

Healthcare makes the limits of aggregate accuracy painfully concrete, because a difference in output by race, ethnicity, or language becomes a difference in diagnosis, triage, and treatment. Zack and colleagues (2024), writing in *The Lancet Digital Health*, reported that GPT-4 could stereotype and fail to represent demographic diversity appropriately when generating clinical vignettes. Using real patient cases, [Yang and colleagues (2024)](https://doi.org/10.1038/s43856-024-00601-z) found that LLM-generated predictions for hospitalization, cost, and mortality shifted when race was varied even with the same case facts. A related vulnerability compounds the risk: a 2026 study reported by [Reuters](https://www.reuters.com/business/healthcare-pharmaceuticals/medical-misinformation-more-likely-fool-ai-if-source-appears-legitimate-study-2026-02-09/) found that models were more likely to accept misinformation when it arrived in authoritative-looking clinical documents, which is precisely how real medical information is exchanged.

### Hiring: résumé screening and bias presented as a ranking

Hiring is where a fairness failure turns into a livelihood outcome, and the modern pattern is usually a ranked list or an auto-summary rather than an outright rejection. [Wilson and Caliskan (2024)](https://arxiv.org/abs/2407.20371) studied résumé screening via language-model retrieval and found gender, race, and intersectional bias in how the tools rank candidates, with white-associated names favoured in many cases. The pattern extends well beyond the categories that dominate Western debates: studying caste, an axis that shapes opportunity for more than a billion people, [Zaveri and Shah (2025)](https://ssrn.com/abstract=5427214) of IIM-Bangalore found that models used for hiring under-represented candidates from marginalized castes, shortlisted dominant-caste names more often, and associated lower-paying, lower-education occupations with the marginalized groups. The mechanism is the familiar one: the model reproduces the hierarchy it absorbed from its training data and presents the result as a neutral ranking.

### Lending: discrimination without a discriminatory author

Financial services add a specific danger, because a model need not be the official underwriting system to influence who gets credit and on what terms. Bowen, Price, Stein, and Yang (2024) evaluated LLM responses to mortgage underwriting tasks using real loan-application data and found that the models recommended more denials and higher interest rates for Black applicants than for otherwise identical white applicants, while also showing that simple prompt interventions could reduce the disparity. This case collapses a common misconception, since discriminatory output can appear even when no explicitly discriminatory content sits in the training data; the bias enters through demographic cues that leak in through names, locations, and narrative context, needing no bigoted author anywhere in the pipeline.

## Where this leaves us

Across religion, dialect, non-native language, gender, caste, and race, the same failure recurs, which is the strongest argument that fairness in generative AI is a structural property we have to instrument rather than a rare defect we can patch after the fact. Teams tend to optimize accuracy, safety, and latency because those are easy to measure, yet accuracy hides subgroup failure, safety can become unequal access when filters trip more often for some identities, and latency scales harm by making a cheap, fast, biased system easy to deploy at population scale. Each of those three is necessary and none is sufficient, which is the gap FAIRBench exists to close by turning the principles above into metrics a team can test, gate, and monitor like any other reliability requirement.

If you want the full technical treatment, the [FAIRBench white paper](whitepaper.md) lays out the architecture, the four fairness dimensions, and the formal metric definitions, while the [Metrics reference](metrics.md) gives thresholds and interpretation and [Reading Your Scorecard](reading-your-scorecard.md) shows how a verdict is assembled. If you would rather start measuring, the [installation guide](getting-started/installation.md) will have you running your first audit in a few minutes.

## References

Rawls, John. *A Theory of Justice*. Harvard University Press, 1971. Overview: [Stanford Encyclopedia of Philosophy — John Rawls](https://plato.stanford.edu/entries/rawls/).

Sen, Amartya. *The Idea of Justice*. Harvard University Press, 2009. Overview: [Stanford Encyclopedia of Philosophy — The Capability Approach](https://plato.stanford.edu/entries/capability-approach/).

Gallegos, I. O., Rossi, R. A., Barrow, J., et al. Bias and Fairness in Large Language Models: A Survey. *Computational Linguistics* 50(3), 2024. <https://doi.org/10.1162/coli_a_00524>

Abid, A., Farooqi, M., and Zou, J. Persistent Anti-Muslim Bias in Large Language Models. *AIES '21*, 2021. <https://doi.org/10.1145/3461702.3462624>

Kotek, H., Dockum, R., and Sun, D. Q. Gender bias and stereotypes in Large Language Models. *CI '23*, 2023. <https://doi.org/10.1145/3582269.3615599>

Hofmann, V., Kalluri, P. R., Jurafsky, D., and King, S. AI generates covertly racist decisions about people based on their dialect. *Nature* 633, 2024. <https://doi.org/10.1038/s41586-024-07856-5>

Liang, W., Yuksekgonul, M., Mao, Y., Wu, E., and Zou, J. GPT detectors are biased against non-native English writers. *Patterns* 4(7), 2023. <https://doi.org/10.1016/j.patter.2023.100779>

Tate, T. P., Steiss, J., Bailey, D., et al. Can AI provide useful holistic essay scoring? *Computers and Education: Artificial Intelligence* 7, 2024. <https://doi.org/10.1016/j.caeai.2024.100255>

Dahl, M., Magesh, V., Suzgun, M., and Ho, D. E. Large Legal Fictions: Profiling Legal Hallucinations in Large Language Models. *Journal of Legal Analysis* 16(1), 2024. <https://doi.org/10.1093/jla/laae003>

*Lexos Media IP LLC v. Overstock.com, Inc.* (D. Kan., 2026). Sanctions order, Feb. 2, 2026, reported in the National Law Review. <https://natlawreview.com/article/judge-issues-public-admonition-12000-sanctions-hallucinations>

Zack, T., Lehman, E., Suzgun, M., et al. Assessing the potential of GPT-4 to perpetuate racial and gender biases in health care. *The Lancet Digital Health* 6, 2024.

Yang, Y., Liu, X., Jin, Q., Huang, F., and Lu, Z. Unmasking and quantifying racial bias of large language models in medical report generation. *Communications Medicine* 4(176), 2024. <https://doi.org/10.1038/s43856-024-00601-z>

Medical misinformation more likely to fool AI if source appears legitimate, study finds. *Reuters*, Feb. 9, 2026. <https://www.reuters.com/business/healthcare-pharmaceuticals/medical-misinformation-more-likely-fool-ai-if-source-appears-legitimate-study-2026-02-09/>

Wilson, K. and Caliskan, A. Gender, Race, and Intersectional Bias in Resume Screening via Language Model Retrieval, 2024. <https://arxiv.org/abs/2407.20371>

Zaveri, J. and Shah, A. Caste and Occupational Identity in Large Language Models, 2025. SSRN. <https://ssrn.com/abstract=5427214>

Bowen, D. E. III, Price, S. M., Stein, L. C. D., and Yang, K. Measuring and Mitigating Racial Disparities in LLMs: Evidence from a Mortgage Underwriting Experiment, 2024.

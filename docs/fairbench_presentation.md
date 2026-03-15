---
theme: seriph
background: https://chatgpt.com/backend-api/estuary/public_content/enc/eyJpZCI6Im1fNjkwM2E5Zjg3ZDcwODE5MWJmYzg3NzU2NTlhMDJkYzg6ZmlsZV8wMDAwMDAwMDUwN2M2MWY1OTZhNjNjOWJlODBkMDk5MyIsInRzIjoiMjAzOTEiLCJwIjoicHlpIiwiY2lkIjoiMSIsInNpZyI6IjQ0NGZlMDgwYTY2OTdkMjNkYTM1MGZmMjAzYTI4NzA0YTJhMGZiMmYyOWJiM2I3NTdjZTdjNWI3MTA1YTMxYzgiLCJ2IjoiMCIsImdpem1vX2lkIjpudWxsLCJjcCI6bnVsbCwibWEiOm51bGx9
title: "FAIRBench"
authors: "Prasanna Vijayanathan, Ranjana Venkataraman"
transition: slide-left
css: |
  .slidev-layout {
    font-family: 'Calibri', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    font-size: 1.3em !important;
    line-height: 1.8 !important;
  }
  .slidev-layout h1 {
    font-family: 'Calibri', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    font-size: 2.2em !important;
    line-height: 1.4 !important;
  }
  .slidev-layout h2 {
    font-family: 'Calibri', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    font-size: 1.6em !important;
    line-height: 1.5 !important;
  }
  .slidev-layout li {
    margin-bottom: 0.8em !important;
  }
  .slidev-layout p {
    line-height: 1.8 !important;
  }
  body, * {
    font-family: 'Calibri', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
  }
---

# FAIRBench
## A **F**airness **A**ssessment Framework for Generative **AI** **R**esponses

---

# Why FAIRBench?

<v-click>
Generative AI influences public narratives and decisions
</v-click>
<br>
<v-click>
Biases in generative outputs are pervasive and subtle
</v-click>
<br>
<v-click>
FAIRBench provides multi-modal, structured fairness evaluation
</v-click>

<!--
Welcome! Today we introduce FAIRBench, a framework to evaluate fairness in generative AI. This presentation will walk through the motivation, design, use cases, and call to action for the community.
-->

---

# The Problem

- <v-click>Most fairness tools are built for classifiers, not generators</v-click>
- <v-click>Generative bias is subtle, complex, and high-dimensional</v-click>
- <v-click>No standard metrics or testbeds exist for generative fairness</v-click>

<!--
Fairlearn and AIF360 are great for classifiers, but generative systems like GPT or DALL·E need different tools. The outputs are open-ended, and the biases show up in nuanced ways. We need new kinds of fairness evaluations.
-->

---

# State of Fairness in Gen AI

<v-clicks>

<div class="bg-gray-700 text-white p-4 rounded-lg mb-4">
<strong>DALL·E 2</strong> generated <strong>97% men</strong> for "CEO" prompts and <strong>98% women</strong> for "nurse" prompts—far exceeding real-world job gender distributions.
<br><em>—Naik & Nushi, AIES 2023</em>
</div>

<div class="bg-gray-700 text-white p-4 rounded-lg mb-4">
Text-to-image models <strong>exaggerate gender-role biases</strong> even beyond training data: professions associated with men showed even fewer women in outputs.
<br><em>—Girrbach et al., arXiv 2025</em>
</div>

<div class="bg-gray-700 text-white p-4 rounded-lg mb-4">
Ambiguous prompts like "a person at work" almost always generated <strong>Western male figures</strong>, ignoring cultural or gender variation.
<br><em>—Girrbach et al., arXiv 2025</em>
</div>

<div class="bg-gray-700 text-white p-4 rounded-lg mb-4">
<strong>GPT-3</strong> associated the word "Muslim" with violence in <strong>66% of completions</strong>, vs. 20% for "Christian."
<br><em>—Abid et al., Nature Machine Intelligence 2021</em>
</div>

</v-clicks>

---

# Our Approach

- <v-click>FAIRBench is a modular, extensible benchmark</v-click>
- <v-click>Evaluates:
  - Representational Fairness
  - Distributional Fairness
  - Interactional Fairness
  - Procedural Fairness</v-click>
- <v-click>Supports Text, Image, Audio, Code outputs</v-click>

<!--
FAIRBench doesn’t just look at model outputs in one way—it evaluates how models represent different groups, distribute benefits, interact with users, and make decisions.
-->

---

# Key Metrics

| Metric | Description | Fairness Type |
|--------|-------------|----------------|
| RSI | Representation Skew Index | Representational |
| SAR | Stereotype Amplification Ratio | Representational |
| CDS | Counterfactual Divergence Score | Interactional/Procedural |
| ODE | Output Diversity Entropy | Distributional |
| HSI | Harm Severity Index | All Types |

<!--
These metrics allow us to quantify fairness across multiple angles. For example, RSI tells us whether certain identities are over- or underrepresented. CDS shows if changing a sensitive attribute causes unjustified changes in output.
-->

---

# Architecture

![FAIRBench Architecture](/Users/pv/truor/FAIRBench_Architecture.png)

<!--
We start by creating test cases, then generate controlled variations. We run the model and evaluate its outputs, compute metrics, and produce a fairness report.
-->

---

# Use Cases

- <v-click>**Text:** Bias in biographies or news headlines</v-click>
- <v-click>**Image:** Skewed portrayals in occupations or beauty</v-click>
- <v-click>**Audio:** Accent-based variation in tone or politeness</v-click>
- <v-click>**Agents:** Dialogue behavior varies across demographic prompts</v-click>

<!--
Imagine a story generator that always makes CEOs male, or a voice assistant that is less helpful to certain accents. FAIRBench catches such patterns with targeted scenarios and quantitative analysis.
-->

---

# Implementation Phases

1. <v-click>Text-based MVP</v-click>
2. <v-click>Multi-modal extensions (image, audio)</v-click>
3. <v-click>Agentic behavior & interactivity</v-click>
4. <v-click>Community benchmarking and governance</v-click>

<!--
We start with NLP because it’s mature and rich with tools. Then we expand to image, audio, and code. Over time, we want FAIRBench to be a shared community resource with contributions from all corners of AI research and policy.
-->

---

# Policy & Standards Alignment

- <v-click>EU AI Act Article 10 (Data & Model Fairness)</v-click>
- <v-click>NIST AI RMF (2023)</v-click>
- <v-click>IEEE AI Standards: P7003, P7015, P3642.</v-click>
- <v-click>Model & Dataset Cards, Transparency Reports</v-click>

<!--
FAIRBench aligns with real-world compliance needs. Whether it's the EU AI Act, NIST guidance, or upcoming IEEE standards, FAIRBench helps organizations demonstrate fairness in a rigorous, transparent way.
-->

---

# Call to Action

- <v-click>Researchers: Propose metrics, test prompts</v-click>
- <v-click>Engineers: Integrate FAIRBench into evaluation workflows</v-click>
- <v-click>Policymakers: Use FAIRBench for audits & norms</v-click>
- <v-click>Join: Contribute, test, and guide FAIRBench development</v-click>

<!--
We want FAIRBench to be shaped by everyone it affects. Share scenarios, propose metrics, apply it in your workflows, or use it in policy audits. Let’s build a shared benchmark for fairness in AI.
-->

---

# Thank You

<img src="/Users/pv/truor/LI.JPG" style="width: 300px; height: 300px;" />


<br><br>
<span style="color: #666; font-size: 0.9em; font-style: italic;">Let's build a fair and just AI future.</span>

<!--
Thanks for your time and interest. We hope FAIRBench becomes a meaningful step toward more fair, transparent, and accountable generative AI systems.
-->
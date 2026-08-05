"""Evaluation components for FAIRBench.

Three-layer evaluation stack per the FAIRBench specification:

Layer 1 — Deterministic classifiers (always on, no LLM judge):
  DemographicClassifier  pronoun + name-origin signals (RSI, ODE, CDS)
  RefusalClassifier      rule-based refusal detection (DSI)
  ToxicityEvaluator      Detoxify / local model (HSI)
  SentimentEvaluator     sentiment divergence (CDS)
  EmbeddingEvaluator     semantic similarity (CDS, ODE)

Layer 2 — Calibrated LLM judges (opt-in, requires explicit configuration):
  LLMJudgeEvaluator      multi-model judge with spec prompt structure
  JudgeModel             judge model configuration

Layer 3 — Human review triage (always on, post-evaluation):
  TriageRouter           flags outputs for human review
  TriageFlag             individual triage flag
"""

from fairbench_genai.evaluation.base import Evaluator
from fairbench_genai.evaluation.demographic import DemographicClassifier
from fairbench_genai.evaluation.embeddings import EmbeddingEvaluator
from fairbench_genai.evaluation.llm_judge import JudgeModel, LLMJudgeEvaluator
from fairbench_genai.evaluation.pipeline import EvaluationPipeline
from fairbench_genai.evaluation.refusal import RefusalClassifier
from fairbench_genai.evaluation.sentiment import SentimentEvaluator
from fairbench_genai.evaluation.toxicity import ToxicityEvaluator
from fairbench_genai.evaluation.triage import TriageFlag, TriageRouter

__all__ = [
    # Base
    "Evaluator",
    # Layer 1
    "DemographicClassifier",
    "EmbeddingEvaluator",
    "RefusalClassifier",
    "SentimentEvaluator",
    "ToxicityEvaluator",
    # Layer 2
    "JudgeModel",
    "LLMJudgeEvaluator",
    # Layer 3
    "TriageFlag",
    "TriageRouter",
    # Pipeline
    "EvaluationPipeline",
]

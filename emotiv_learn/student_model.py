from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
import re
from typing import Any


ACTION_STYLE_KEYS = {
    "no_change": "same_style",
    "simplify": "accessible",
    "deepen": "technical_depth",
    "summarize": "concise",
    "highlight_key_points": "structured",
    "worked_example": "worked_examples",
    "analogy": "analogies",
    "step_by_step": "step_by_step",
}


@dataclass(frozen=True)
class KnowledgeState:
    concept_mastery: dict[str, float]
    misconceptions: dict[str, float]
    confidence: float
    curiosity: float
    preferred_style: dict[str, float]


@dataclass(frozen=True)
class NeuroState:
    workload: float
    fatigue: float
    attention: float
    vigilance: float
    stress: float
    engagement: float


@dataclass(frozen=True)
class HiddenKnowledgeState:
    knowledge_state: KnowledgeState
    neuro_state: NeuroState

    @property
    def concept_mastery(self) -> dict[str, float]:
        return self.knowledge_state.concept_mastery

    @property
    def misconceptions(self) -> dict[str, float]:
        return self.knowledge_state.misconceptions

    @property
    def confidence(self) -> float:
        return self.knowledge_state.confidence

    @property
    def curiosity(self) -> float:
        return self.knowledge_state.curiosity

    @property
    def preferred_style(self) -> dict[str, float]:
        return self.knowledge_state.preferred_style

    @property
    def workload(self) -> float:
        return self.neuro_state.workload

    @property
    def fatigue(self) -> float:
        return self.neuro_state.fatigue

    @property
    def attention(self) -> float:
        return self.neuro_state.attention

    @property
    def vigilance(self) -> float:
        return self.neuro_state.vigilance

    @property
    def stress(self) -> float:
        return self.neuro_state.stress

    @property
    def engagement(self) -> float:
        return self.neuro_state.engagement


@dataclass(frozen=True)
class StudentTransition:
    hidden_state_before: HiddenKnowledgeState
    hidden_state_after: HiddenKnowledgeState
    evaluation: dict[str, float]
    observable_signals: dict[str, Any]
    response_type_probs: dict[str, float]
    sampled_response_type: str
    oracle_mastery_gain: float
    checkpoint_answer: str | None
    checkpoint_correct: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden_state_before": asdict(self.hidden_state_before),
            "hidden_state_after": asdict(self.hidden_state_after),
            "evaluation": self.evaluation,
            "observable_signals": self.observable_signals,
            "response_type_probs": self.response_type_probs,
            "sampled_response_type": self.sampled_response_type,
            "oracle_mastery_gain": self.oracle_mastery_gain,
            "checkpoint_answer": self.checkpoint_answer,
            "checkpoint_correct": self.checkpoint_correct,
        }


class HiddenKnowledgeStudent:
    """Split-latent simulator with separate knowledge and neurocognitive state."""

    def __init__(self, hidden_state: HiddenKnowledgeState, seed: int = 0, temperature: float = 0.65) -> None:
        self.hidden_state = hidden_state
        self.rng = random.Random(seed)
        self.temperature = temperature

    def step(
        self,
        *,
        concept_id: str,
        action_id: str,
        tutor_message: str,
        checkpoint_expected: bool,
    ) -> StudentTransition:
        before = self.hidden_state
        evaluation = self.evaluate_tutor_message(
            hidden_state=before,
            concept_id=concept_id,
            action_id=action_id,
            tutor_message=tutor_message,
            checkpoint_expected=checkpoint_expected,
        )
        after = self._next_hidden_state(
            hidden_state=before,
            concept_id=concept_id,
            action_id=action_id,
            evaluation=evaluation,
            checkpoint_expected=checkpoint_expected,
        )
        observables = self._behavior_observables(
            hidden_state=after,
            concept_id=concept_id,
            evaluation=evaluation,
            checkpoint_expected=checkpoint_expected,
        )
        response_type_probs = self._response_type_probs(after, concept_id, observables)
        sampled_response_type = self._sample_response_type(response_type_probs)
        checkpoint_correct = observables["checkpoint_correct"]
        checkpoint_answer = self._checkpoint_answer(concept_id, checkpoint_correct, after) if checkpoint_expected else None

        transition = StudentTransition(
            hidden_state_before=before,
            hidden_state_after=after,
            evaluation=evaluation,
            observable_signals=observables,
            response_type_probs=response_type_probs,
            sampled_response_type=sampled_response_type,
            oracle_mastery_gain=(
                after.knowledge_state.concept_mastery[concept_id]
                - before.knowledge_state.concept_mastery.get(concept_id, 0.0)
            ),
            checkpoint_answer=checkpoint_answer,
            checkpoint_correct=checkpoint_correct,
        )
        self.hidden_state = after
        return transition

    def evaluate_tutor_message(
        self,
        *,
        hidden_state: HiddenKnowledgeState,
        concept_id: str,
        action_id: str,
        tutor_message: str,
        checkpoint_expected: bool,
    ) -> dict[str, float]:
        mastery = hidden_state.knowledge_state.concept_mastery.get(concept_id, 0.0)
        text_features = _text_features(tutor_message)
        action_adherence = _action_adherence(action_id, text_features)
        style_fit = hidden_state.knowledge_state.preferred_style.get(ACTION_STYLE_KEYS.get(action_id, "same_style"), 0.5)
        complexity = text_features["complexity"]
        target_complexity = _clip01(0.24 + 0.58 * mastery + 0.08 * hidden_state.neuro_state.attention)
        difficulty_fit = 1.0 - min(abs(complexity - target_complexity) / 0.75, 1.0)
        semantic_friction = _clip01(
            0.55 * (1.0 - mastery)
            + 0.25 * sum(
                value for key, value in hidden_state.knowledge_state.misconceptions.items() if concept_id in key
            )
            + 0.20 * text_features["technical_density"]
        )
        overload = _clip01(
            0.34 * hidden_state.neuro_state.workload
            + 0.20 * hidden_state.neuro_state.fatigue
            + 0.16 * hidden_state.neuro_state.stress
            + 0.14 * complexity
            + 0.10 * semantic_friction
            - 0.12 * hidden_state.neuro_state.attention
        )
        clarity = _clip01(
            0.30
            + 0.28 * text_features["structure"]
            + 0.22 * action_adherence
            + 0.14 * text_features["supportive"]
            - 0.22 * overload
        )
        novelty = _clip01(
            0.22
            + 0.32 * action_adherence
            + 0.20 * text_features["example_density"]
            + 0.16 * int(action_id in {"deepen", "analogy", "worked_example"})
            - 0.22 * mastery
        )
        supportiveness = _clip01(
            0.38
            + 0.22 * text_features["supportive"]
            + 0.18 * clarity
            + 0.10 * text_features["structure"]
            - 0.18 * overload
        )
        checkpoint_quality = (
            _clip01(text_features["checkpoint_present"]) if checkpoint_expected else 1.0 - text_features["checkpoint_present"]
        )

        return {
            "action_adherence": action_adherence,
            "style_fit": _clip01(style_fit),
            "difficulty_fit": _clip01(difficulty_fit),
            "clarity": clarity,
            "novelty": novelty,
            "overload": overload,
            "supportiveness": supportiveness,
            "checkpoint_quality": _clip01(checkpoint_quality),
            "message_complexity": complexity,
            "semantic_friction": semantic_friction,
        }

    def _next_hidden_state(
        self,
        *,
        hidden_state: HiddenKnowledgeState,
        concept_id: str,
        action_id: str,
        evaluation: dict[str, float],
        checkpoint_expected: bool,
    ) -> HiddenKnowledgeState:
        knowledge_before = hidden_state.knowledge_state
        neuro_before = hidden_state.neuro_state
        mastery_before = knowledge_before.concept_mastery.get(concept_id, 0.0)

        instructional_effect = (
            0.32 * evaluation["action_adherence"]
            + 0.24 * evaluation["style_fit"]
            + 0.22 * evaluation["difficulty_fit"]
            + 0.22 * evaluation["clarity"]
        )
        room_to_learn = 1.0 - mastery_before
        boredom = _clip01((mastery_before - 0.72) * (1.0 - evaluation["novelty"]))
        mastery_delta = 0.16 * instructional_effect * room_to_learn
        mastery_delta += 0.04 * neuro_before.attention
        mastery_delta += 0.03 * neuro_before.vigilance
        mastery_delta -= 0.08 * neuro_before.workload * (1.0 - evaluation["supportiveness"])
        mastery_delta -= 0.05 * boredom
        if checkpoint_expected:
            mastery_delta += 0.04 * evaluation["checkpoint_quality"]
        mastery_after = _clip01(mastery_before + mastery_delta)

        concept_mastery = dict(knowledge_before.concept_mastery)
        concept_mastery[concept_id] = mastery_after

        misconceptions = dict(knowledge_before.misconceptions)
        repair_strength = 0.11 * evaluation["clarity"] * evaluation["difficulty_fit"] * (1.0 - neuro_before.workload)
        for key, value in misconceptions.items():
            if concept_id in key:
                misconceptions[key] = _clip01(value - repair_strength)

        confidence = _clip01(
            knowledge_before.confidence
            + 0.18 * (mastery_after - mastery_before)
            + 0.05 * evaluation["clarity"]
            - 0.07 * neuro_before.stress
            - 0.06 * neuro_before.workload
        )
        curiosity = _clip01(
            knowledge_before.curiosity
            + 0.08 * evaluation["novelty"]
            + 0.04 * int(action_id in {"deepen", "analogy", "worked_example"})
            - 0.05 * neuro_before.fatigue
            - 0.04 * boredom
        )

        workload = _clip01(
            0.54 * neuro_before.workload
            + 0.20 * evaluation["message_complexity"]
            + 0.14 * evaluation["semantic_friction"]
            + 0.08 * int(action_id == "deepen")
            - 0.16 * evaluation["supportiveness"]
            - 0.08 * int(action_id in {"summarize", "highlight_key_points"})
        )
        fatigue = _clip01(
            0.70 * neuro_before.fatigue
            + 0.15 * workload
            + 0.08 * boredom
            + 0.05 * int(action_id == "deepen")
            - 0.08 * evaluation["supportiveness"]
        )
        attention = _clip01(
            0.58 * neuro_before.attention
            + 0.20 * evaluation["clarity"]
            + 0.12 * evaluation["supportiveness"]
            - 0.16 * workload
            - 0.08 * fatigue
        )
        vigilance = _clip01(
            0.64 * neuro_before.vigilance
            + 0.10 * evaluation["novelty"]
            + 0.08 * attention
            - 0.10 * fatigue
            - 0.08 * workload
        )
        stress = _clip01(
            0.55 * neuro_before.stress
            + 0.20 * workload
            + 0.12 * evaluation["semantic_friction"]
            - 0.12 * evaluation["supportiveness"]
            - 0.06 * knowledge_before.confidence
        )
        engagement = _clip01(
            0.52 * neuro_before.engagement
            + 0.18 * evaluation["supportiveness"]
            + 0.14 * evaluation["novelty"]
            + 0.08 * curiosity
            - 0.12 * fatigue
            - 0.10 * workload
        )

        return HiddenKnowledgeState(
            knowledge_state=KnowledgeState(
                concept_mastery=concept_mastery,
                misconceptions=misconceptions,
                confidence=confidence,
                curiosity=curiosity,
                preferred_style=dict(knowledge_before.preferred_style),
            ),
            neuro_state=NeuroState(
                workload=workload,
                fatigue=fatigue,
                attention=attention,
                vigilance=vigilance,
                stress=stress,
                engagement=engagement,
            ),
        )

    def _behavior_observables(
        self,
        *,
        hidden_state: HiddenKnowledgeState,
        concept_id: str,
        evaluation: dict[str, float],
        checkpoint_expected: bool,
    ) -> dict[str, Any]:
        knowledge = hidden_state.knowledge_state
        neuro = hidden_state.neuro_state
        mastery = knowledge.concept_mastery.get(concept_id, 0.0)
        misconception_mass = sum(value for key, value in knowledge.misconceptions.items() if concept_id in key)

        confusion = _clip01(
            0.45 * (1.0 - mastery)
            + 0.20 * misconception_mass
            + 0.18 * neuro.workload
            + 0.10 * neuro.stress
            - 0.12 * knowledge.confidence
            - 0.06 * evaluation["clarity"]
        )
        comprehension = _clip01(
            0.56 * mastery
            + 0.14 * evaluation["clarity"]
            + 0.10 * knowledge.confidence
            + 0.08 * neuro.attention
            - 0.12 * misconception_mass
            - 0.10 * neuro.workload
        )
        progress = _clip01(
            0.46 * mastery
            + 0.16 * evaluation["clarity"]
            + 0.14 * neuro.attention
            + 0.08 * neuro.vigilance
            - 0.12 * neuro.workload
            - 0.06 * neuro.fatigue
        )
        checkpoint_probability = _clip01(
            0.16
            + 0.48 * mastery
            + 0.12 * knowledge.confidence
            + 0.10 * neuro.attention
            - 0.14 * confusion
            - 0.08 * misconception_mass
        )
        checkpoint_correct = None
        if checkpoint_expected:
            checkpoint_correct = self.rng.random() < checkpoint_probability

        return {
            "followup_type": "unknown",
            "checkpoint_correct": checkpoint_correct,
            "checkpoint_score": checkpoint_probability if checkpoint_expected else None,
            "confusion_score": confusion,
            "comprehension_score": comprehension,
            "engagement_score": neuro.engagement,
            "progress_signal": progress,
            "pace_fast_score": _clip01(0.50 * mastery + 0.20 * neuro.vigilance - 0.35 * neuro.workload - 0.20 * neuro.fatigue),
            "pace_slow_score": _clip01(0.42 * confusion + 0.24 * neuro.workload + 0.20 * neuro.fatigue - 0.12 * neuro.attention),
            "confidence": knowledge.confidence,
            "attention": neuro.attention,
            "fatigue": neuro.fatigue,
            "workload": neuro.workload,
            "vigilance": neuro.vigilance,
            "stress": neuro.stress,
            "semantic_friction": evaluation["semantic_friction"],
        }

    def _response_type_probs(
        self,
        hidden_state: HiddenKnowledgeState,
        concept_id: str,
        observables: dict[str, Any],
    ) -> dict[str, float]:
        knowledge = hidden_state.knowledge_state
        neuro = hidden_state.neuro_state
        mastery = knowledge.concept_mastery.get(concept_id, 0.0)
        confusion = observables["confusion_score"]

        logits = {
            "continue": (
                1.08 * mastery
                + 0.58 * knowledge.confidence
                + 0.28 * neuro.attention
                - 0.95 * confusion
                - 0.42 * neuro.fatigue
                - 0.26 * neuro.workload
            ),
            "clarify": (
                1.45 * confusion
                + 0.30 * neuro.stress
                + 0.28 * neuro.workload
                - 0.86 * mastery
                - 0.34 * knowledge.confidence
            ),
            "branch": (
                0.96 * knowledge.curiosity
                + 0.34 * neuro.engagement
                + 0.18 * mastery
                + 0.12 * neuro.vigilance
                - 0.58 * confusion
                - 0.22 * neuro.fatigue
            ),
        }
        return _softmax(logits, self.temperature)

    def _sample_response_type(self, probs: dict[str, float]) -> str:
        threshold = self.rng.random()
        cumulative = 0.0
        for response_type, probability in probs.items():
            cumulative += probability
            if threshold <= cumulative:
                return response_type
        return "continue"

    def _checkpoint_answer(
        self,
        concept_id: str,
        checkpoint_correct: bool | None,
        hidden_state: HiddenKnowledgeState,
    ) -> str:
        if checkpoint_correct:
            return f"I think {concept_id} means using the main idea correctly in the current step."
        if hidden_state.knowledge_state.confidence < 0.35:
            return "I'm not sure yet; I can only guess at the relationship."
        return "I have a partial idea, but I may be mixing up the pieces."


def default_hidden_knowledge_state() -> HiddenKnowledgeState:
    return HiddenKnowledgeState(
        knowledge_state=KnowledgeState(
            concept_mastery={
                "gradient": 0.45,
                "learning_rate": 0.35,
                "gradient_descent_update": 0.25,
                "overshooting": 0.30,
                "convergence": 0.28,
            },
            misconceptions={
                "gradient_descent_update_sign": 0.45,
                "learning_rate_bigger_is_always_better": 0.55,
                "convergence_equals_one_step": 0.40,
            },
            confidence=0.50,
            curiosity=0.55,
            preferred_style={
                "same_style": 0.50,
                "accessible": 0.60,
                "technical_depth": 0.45,
                "concise": 0.70,
                "structured": 0.75,
                "worked_examples": 0.85,
                "analogies": 0.65,
                "step_by_step": 0.80,
            },
        ),
        neuro_state=NeuroState(
            workload=0.34,
            fatigue=0.20,
            attention=0.70,
            vigilance=0.66,
            stress=0.22,
            engagement=0.70,
        ),
    )


def _text_features(text: str) -> dict[str, float]:
    words = re.findall(r"[A-Za-z']+", text.lower())
    sentences = max(len(re.findall(r"[.!?]", text)), 1)
    word_count = len(words)
    avg_sentence_len = word_count / sentences
    unique_ratio = len(set(words)) / max(word_count, 1)
    bullet_count = len(re.findall(r"(^|\n)\s*(-|\*|\d+\.)\s+", text))
    question_count = text.count("?")
    technical_terms = sum(
        1
        for word in words
        if word
        in {
            "gradient",
            "derivative",
            "parameter",
            "loss",
            "optimize",
            "convergence",
            "learning",
            "rate",
            "update",
            "minimum",
        }
    )
    example_terms = sum(1 for word in words if word in {"example", "imagine", "suppose", "let", "step"})
    supportive_terms = sum(1 for word in words if word in {"try", "think", "help", "notice", "remember"})

    complexity = _clip01(0.35 * (avg_sentence_len / 26.0) + 0.35 * unique_ratio + 0.30 * (technical_terms / 10.0))
    structure = _clip01(0.25 + 0.15 * min(bullet_count, 4) + 0.10 * min(example_terms, 4))
    return {
        "word_count": float(word_count),
        "complexity": complexity,
        "structure": structure,
        "technical_density": _clip01(technical_terms / max(word_count / 20.0, 1.0)),
        "example_density": _clip01(example_terms / max(word_count / 30.0, 1.0)),
        "supportive": _clip01(supportive_terms / 4.0),
        "checkpoint_present": 1.0 if question_count else 0.0,
    }


def _action_adherence(action_id: str, text_features: dict[str, float]) -> float:
    word_count = text_features["word_count"]
    if action_id == "no_change":
        return 0.55
    if action_id == "simplify":
        return _clip01(0.65 + 0.30 * (1.0 - text_features["complexity"]) - 0.20 * (word_count > 180))
    if action_id == "deepen":
        return _clip01(0.45 + 0.45 * text_features["technical_density"] + 0.20 * text_features["complexity"])
    if action_id == "summarize":
        return _clip01(0.90 - max(word_count - 130.0, 0.0) / 180.0)
    if action_id == "highlight_key_points":
        return _clip01(0.35 + 0.65 * text_features["structure"])
    if action_id == "worked_example":
        return _clip01(0.30 + 0.70 * text_features["example_density"])
    if action_id == "analogy":
        return _clip01(0.35 + 0.45 * text_features["example_density"] + 0.20 * (1.0 - text_features["technical_density"]))
    if action_id == "step_by_step":
        return _clip01(0.30 + 0.70 * text_features["structure"])
    return 0.5


def _softmax(logits: dict[str, float], temperature: float) -> dict[str, float]:
    adjusted = {key: value / max(temperature, 0.01) for key, value in logits.items()}
    max_logit = max(adjusted.values())
    exp_values = {key: math.exp(value - max_logit) for key, value in adjusted.items()}
    total = sum(exp_values.values())
    return {key: value / total for key, value in exp_values.items()}


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

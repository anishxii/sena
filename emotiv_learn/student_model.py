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
class HiddenKnowledgeState:
    concept_mastery: dict[str, float]
    misconceptions: dict[str, float]
    fatigue: float
    confidence: float
    curiosity: float
    attention: float
    engagement: float
    preferred_style: dict[str, float]


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
    """Simulator-only learner model with hidden concept mastery and observable emissions."""

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
        observables = self._observable_signals(
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
            oracle_mastery_gain=after.concept_mastery[concept_id] - before.concept_mastery.get(concept_id, 0.0),
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
        mastery = hidden_state.concept_mastery.get(concept_id, 0.0)
        text_features = _text_features(tutor_message)
        action_adherence = _action_adherence(action_id, text_features)
        style_fit = hidden_state.preferred_style.get(ACTION_STYLE_KEYS.get(action_id, "same_style"), 0.5)
        complexity = text_features["complexity"]
        target_complexity = 0.30 + 0.55 * mastery
        difficulty_fit = 1.0 - min(abs(complexity - target_complexity) / 0.70, 1.0)
        overload = _clip01(0.55 * complexity + 0.35 * hidden_state.fatigue - 0.45 * mastery)
        clarity = _clip01(0.35 + 0.25 * text_features["structure"] + 0.25 * action_adherence - 0.25 * overload)
        novelty = _clip01(0.25 + 0.40 * action_adherence + 0.20 * complexity - 0.35 * mastery)
        supportiveness = _clip01(0.45 + 0.20 * text_features["supportive"] + 0.20 * clarity - 0.15 * overload)
        checkpoint_quality = _clip01(text_features["checkpoint_present"]) if checkpoint_expected else 1.0 - text_features["checkpoint_present"]

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
        mastery_before = hidden_state.concept_mastery.get(concept_id, 0.0)
        instructional_effect = (
            0.30 * evaluation["action_adherence"]
            + 0.25 * evaluation["style_fit"]
            + 0.25 * evaluation["difficulty_fit"]
            + 0.20 * evaluation["clarity"]
        )
        room_to_learn = 1.0 - mastery_before
        boredom = _clip01((mastery_before - 0.70) * (1.0 - evaluation["novelty"]))
        mastery_delta = 0.18 * instructional_effect * room_to_learn
        mastery_delta -= 0.06 * evaluation["overload"] * (1.0 - evaluation["supportiveness"])
        mastery_delta -= 0.03 * boredom
        if checkpoint_expected:
            mastery_delta += 0.04 * evaluation["checkpoint_quality"]
        mastery_after = _clip01(mastery_before + mastery_delta)

        concept_mastery = dict(hidden_state.concept_mastery)
        concept_mastery[concept_id] = mastery_after

        attention = _clip01(hidden_state.attention + 0.06 * evaluation["clarity"] - 0.08 * evaluation["overload"])
        confidence = _clip01(
            hidden_state.confidence
            + 0.16 * (mastery_after - mastery_before)
            + 0.05 * evaluation["clarity"]
            - 0.09 * evaluation["overload"]
        )
        curiosity = _clip01(
            hidden_state.curiosity
            + 0.08 * evaluation["novelty"]
            + 0.03 * int(action_id in {"deepen", "analogy"})
            - 0.05 * evaluation["overload"]
            - 0.05 * int(mastery_after > 0.85)
        )
        fatigue = _clip01(
            hidden_state.fatigue
            + 0.04 * evaluation["overload"]
            + 0.03 * boredom
            - 0.03 * evaluation["supportiveness"]
        )
        engagement = _clip01(
            hidden_state.engagement
            + 0.07 * evaluation["supportiveness"]
            + 0.05 * evaluation["novelty"]
            - 0.08 * fatigue
            - 0.05 * evaluation["overload"]
        )

        misconceptions = dict(hidden_state.misconceptions)
        for key, value in misconceptions.items():
            if concept_id in key:
                misconceptions[key] = _clip01(value - 0.12 * evaluation["clarity"] * evaluation["difficulty_fit"])

        return HiddenKnowledgeState(
            concept_mastery=concept_mastery,
            misconceptions=misconceptions,
            fatigue=fatigue,
            confidence=confidence,
            curiosity=curiosity,
            attention=attention,
            engagement=engagement,
            preferred_style=dict(hidden_state.preferred_style),
        )

    def _observable_signals(
        self,
        *,
        hidden_state: HiddenKnowledgeState,
        concept_id: str,
        evaluation: dict[str, float],
        checkpoint_expected: bool,
    ) -> dict[str, Any]:
        mastery = hidden_state.concept_mastery.get(concept_id, 0.0)
        confusion = _clip01(0.70 * (1.0 - mastery) + 0.35 * evaluation["overload"] - 0.25 * hidden_state.confidence)
        comprehension = _clip01(0.70 * mastery + 0.20 * evaluation["clarity"] + 0.10 * hidden_state.confidence)
        progress = _clip01(0.65 * evaluation["clarity"] * (1.0 - evaluation["overload"]) + 0.35 * mastery)
        checkpoint_probability = _clip01(0.15 + 0.70 * mastery + 0.10 * hidden_state.confidence - 0.25 * confusion)
        checkpoint_correct = None
        if checkpoint_expected:
            checkpoint_correct = self.rng.random() < checkpoint_probability

        return {
            "followup_type": "unknown",
            "checkpoint_correct": checkpoint_correct,
            "checkpoint_score": checkpoint_probability if checkpoint_expected else None,
            "confusion_score": confusion,
            "comprehension_score": comprehension,
            "engagement_score": hidden_state.engagement,
            "progress_signal": progress,
            "pace_fast_score": _clip01(mastery - 0.15 - hidden_state.fatigue),
            "pace_slow_score": _clip01(confusion + hidden_state.fatigue - 0.65),
            "confidence": hidden_state.confidence,
            "attention": hidden_state.attention,
            "fatigue": hidden_state.fatigue,
        }

    def _response_type_probs(
        self,
        hidden_state: HiddenKnowledgeState,
        concept_id: str,
        observables: dict[str, Any],
    ) -> dict[str, float]:
        mastery = hidden_state.concept_mastery.get(concept_id, 0.0)
        confusion = observables["confusion_score"]
        fatigue = hidden_state.fatigue
        confidence = hidden_state.confidence
        curiosity = hidden_state.curiosity
        engagement = hidden_state.engagement

        logits = {
            "continue": 1.25 * mastery + 0.75 * confidence - 1.15 * confusion - 0.45 * fatigue,
            "clarify": 1.55 * confusion + 0.45 * fatigue - 0.95 * mastery - 0.35 * confidence,
            "branch": 1.10 * curiosity + 0.45 * engagement + 0.20 * mastery - 0.75 * confusion - 0.25 * fatigue,
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
        if hidden_state.confidence < 0.35:
            return "I'm not sure yet; I can only guess at the relationship."
        return "I have a partial idea, but I may be mixing up the pieces."


def default_hidden_knowledge_state() -> HiddenKnowledgeState:
    return HiddenKnowledgeState(
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
        fatigue=0.20,
        confidence=0.50,
        curiosity=0.55,
        attention=0.70,
        engagement=0.70,
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

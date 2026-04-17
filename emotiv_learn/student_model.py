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
    claims_understood: dict[str, list[str]]
    misconceptions: dict[str, float]
    confidence: float
    curiosity: float
    preferred_style: dict[str, float]
    learning_style: dict[str, float]


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
    def claims_understood(self) -> dict[str, list[str]]:
        return self.knowledge_state.claims_understood

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
    def learning_style(self) -> dict[str, float]:
        return self.knowledge_state.learning_style

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
            tutor_message=tutor_message,
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
        content_mix = _content_mix(text_features=text_features, action_id=action_id)
        action_adherence = _action_adherence(action_id, text_features)
        preferred_style_fit = hidden_state.knowledge_state.preferred_style.get(
            ACTION_STYLE_KEYS.get(action_id, "same_style"),
            0.5,
        )
        learning_style_fit = _learning_style_fit(
            learning_style=hidden_state.knowledge_state.learning_style,
            text_features=text_features,
            action_id=action_id,
        )
        style_fit = _clip01(0.60 * preferred_style_fit + 0.40 * learning_style_fit)
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
            "content_text_description": content_mix["text_description"],
            "content_text_examples": content_mix["text_examples"],
            "content_visual": content_mix["visual"],
            "visual_structure": text_features["visual_structure"],
        }

    def _next_hidden_state(
        self,
        *,
        hidden_state: HiddenKnowledgeState,
        concept_id: str,
        action_id: str,
        tutor_message: str,
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
        claims_understood = _updated_claims_understood(
            claims_before=knowledge_before.claims_understood,
            concept_id=concept_id,
            tutor_message=tutor_message,
            mastery_before=mastery_before,
            mastery_after=mastery_after,
            clarity=evaluation["clarity"],
            checkpoint_expected=checkpoint_expected,
            checkpoint_quality=evaluation["checkpoint_quality"],
        )

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
                claims_understood=claims_understood,
                misconceptions=misconceptions,
                confidence=confidence,
                curiosity=curiosity,
                preferred_style=dict(knowledge_before.preferred_style),
                learning_style=dict(knowledge_before.learning_style),
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
                "loss_function": 0.38,
                "prediction_error": 0.34,
                "parameter": 0.36,
                "weight_initialization": 0.26,
                "activation_function": 0.30,
                "neuron_output": 0.32,
                "partial_derivative": 0.24,
                "learning_rate": 0.35,
                "gradient_descent_update": 0.25,
                "batch_size": 0.24,
                "stochastic_gradient": 0.22,
                "chain_rule": 0.18,
                "backpropagation": 0.16,
                "hidden_layer_credit_assignment": 0.14,
                "momentum": 0.20,
                "overshooting": 0.30,
                "vanishing_gradient": 0.16,
                "local_minimum": 0.27,
                "curvature": 0.22,
                "regularization": 0.20,
                "convergence": 0.28,
            },
            claims_understood={
                "gradient": [],
                "loss_function": [],
                "prediction_error": [],
                "parameter": [],
                "weight_initialization": [],
                "activation_function": [],
                "neuron_output": [],
                "partial_derivative": [],
                "learning_rate": [],
                "gradient_descent_update": [],
                "batch_size": [],
                "stochastic_gradient": [],
                "chain_rule": [],
                "backpropagation": [],
                "hidden_layer_credit_assignment": [],
                "momentum": [],
                "overshooting": [],
                "vanishing_gradient": [],
                "local_minimum": [],
                "curvature": [],
                "regularization": [],
                "convergence": [],
            },
            misconceptions={
                "gradient_descent_update_sign": 0.45,
                "learning_rate_bigger_is_always_better": 0.55,
                "prediction_error_equals_accuracy": 0.44,
                "activation_function_is_optional_everywhere": 0.39,
                "partial_derivative_changes_all_variables_at_once": 0.43,
                "batch_size_one_equals_full_batch": 0.40,
                "stochastic_gradient_is_noise_only": 0.36,
                "chain_rule_only_applies_once": 0.46,
                "backpropagation_skips_hidden_layers": 0.42,
                "momentum_means_increasing_learning_rate": 0.41,
                "vanishing_gradient_means_no_learning_anywhere": 0.40,
                "local_minimum_always_global": 0.48,
                "regularization_always_improves_training_loss": 0.37,
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
            learning_style={
                "text_description": 0.55,
                "text_examples": 0.35,
                "visual": 0.10,
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
    arrow_count = text.count("->") + text.count("=>")
    label_count = len(re.findall(r"(^|\n)\s*[A-Za-z][A-Za-z ]{0,24}:\s", text))
    line_count = max(len([line for line in text.splitlines() if line.strip()]), 1)
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
        "visual_structure": _clip01(
            0.35 * min(bullet_count, 5) / 5.0
            + 0.25 * min(label_count, 4) / 4.0
            + 0.20 * min(arrow_count, 4) / 4.0
            + 0.20 * min(line_count / max(sentences, 1), 3.0) / 3.0
        ),
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


CANONICAL_CLAIMS = {
    "gradient": [
        "The gradient points in the direction of steepest increase of the function.",
        "To reduce the objective, gradient descent moves opposite the gradient.",
    ],
    "loss_function": [
        "The loss function measures how far the model's predictions are from the target values.",
        "Training tries to reduce loss by changing parameters in directions that improve predictions.",
    ],
    "prediction_error": [
        "Prediction error is the gap between what the model predicts and what the target says should happen.",
        "Loss aggregates prediction error into a quantity optimization can minimize.",
    ],
    "parameter": [
        "Parameters are the tunable weights or values the optimizer updates during learning.",
        "Changing parameters changes the model's predictions and therefore the loss.",
    ],
    "weight_initialization": [
        "Weight initialization sets the starting parameter values before training begins.",
        "Initialization affects gradient flow and how quickly optimization can make progress.",
    ],
    "activation_function": [
        "An activation function introduces nonlinearity so layered networks can model complex patterns.",
        "Without nonlinear activations, stacked linear layers collapse into another linear transformation.",
    ],
    "neuron_output": [
        "A neuron output is the activation produced after combining inputs with weights and applying the activation function.",
        "Neuron outputs become inputs to later layers and ultimately affect the loss.",
    ],
    "partial_derivative": [
        "A partial derivative measures how the loss changes when one variable changes while others are held fixed.",
        "Gradients are collections of partial derivatives for all model parameters.",
    ],
    "learning_rate": [
        "The learning rate controls the size of each parameter update step.",
        "A learning rate that is too large can make optimization unstable.",
    ],
    "gradient_descent_update": [
        "A gradient descent update subtracts learning_rate times gradient from the current parameter.",
        "The update rule uses both the gradient direction and the learning rate magnitude.",
    ],
    "batch_size": [
        "Batch size is the number of training examples used to estimate a gradient in one update.",
        "Different batch sizes trade off gradient stability, memory use, and update frequency.",
    ],
    "stochastic_gradient": [
        "Stochastic gradient methods use small batches or single examples to estimate gradients more frequently.",
        "Noisier gradient estimates can still make progress and sometimes help optimization escape poor regions.",
    ],
    "chain_rule": [
        "The chain rule breaks a composite derivative into local derivatives multiplied along the path.",
        "Backpropagation uses the chain rule to move gradients from the output layer back to earlier layers.",
    ],
    "backpropagation": [
        "Backpropagation computes how each parameter affects loss by propagating gradients backward through the network.",
        "It reuses intermediate derivatives so gradients for all layers can be computed efficiently.",
    ],
    "hidden_layer_credit_assignment": [
        "Hidden-layer credit assignment is the problem of determining how internal units contributed to output error.",
        "Backpropagation solves credit assignment by passing gradients backward to hidden layers.",
    ],
    "momentum": [
        "Momentum accumulates a running direction of past gradients to smooth and accelerate optimization.",
        "Momentum can reduce zig-zagging and help updates move consistently through shallow valleys.",
    ],
    "overshooting": [
        "Overshooting happens when updates are too large and jump past the minimum.",
        "Large learning rates can cause oscillation instead of steady improvement.",
    ],
    "vanishing_gradient": [
        "Vanishing gradients occur when gradient signals shrink as they pass backward through many layers.",
        "Small gradients make deep parameters learn slowly because their updates become tiny.",
    ],
    "local_minimum": [
        "A local minimum is a point where nearby moves increase loss even if a better point may exist elsewhere.",
        "Optimization can slow down or stall near flat regions and local minima depending on the landscape.",
    ],
    "curvature": [
        "Curvature describes how sharply the loss surface bends around a point in parameter space.",
        "High curvature can make a fixed learning rate unstable in some directions and slow in others.",
    ],
    "regularization": [
        "Regularization adds constraints or penalties that encourage simpler models or smaller weights.",
        "It can improve generalization even if it slightly changes the training objective.",
    ],
    "convergence": [
        "Convergence means repeated updates are bringing the parameters toward a stable lower-loss region.",
        "As optimization converges, updates often become smaller and loss changes less dramatically.",
    ],
}

CONCEPT_KEYWORDS = {
    "gradient": {"gradient", "direction", "increase", "steepest"},
    "loss_function": {"loss", "objective", "prediction", "error", "target"},
    "prediction_error": {"prediction", "error", "target", "wrong", "difference"},
    "parameter": {"parameter", "weight", "weights", "value", "model"},
    "weight_initialization": {"initialization", "initial", "start", "weights", "parameters"},
    "activation_function": {"activation", "nonlinear", "relu", "sigmoid", "tanh"},
    "neuron_output": {"neuron", "output", "activation", "signal", "layer"},
    "partial_derivative": {"partial", "derivative", "variable", "fixed", "change"},
    "learning_rate": {"learning", "rate", "step", "size", "update"},
    "gradient_descent_update": {"update", "parameter", "gradient", "learning", "rate", "subtract"},
    "batch_size": {"batch", "examples", "mini-batch", "samples", "update"},
    "stochastic_gradient": {"stochastic", "sample", "mini-batch", "noisy", "gradient"},
    "chain_rule": {"chain", "rule", "derivative", "compose", "product", "path"},
    "backpropagation": {"backpropagation", "backprop", "backward", "layer", "gradient", "network"},
    "hidden_layer_credit_assignment": {"hidden", "credit", "assignment", "layer", "responsibility", "error"},
    "momentum": {"momentum", "velocity", "past", "gradient", "smoothing"},
    "overshooting": {"overshooting", "overshoot", "jump", "past", "minimum", "oscillation"},
    "vanishing_gradient": {"vanishing", "gradient", "tiny", "shrink", "deep", "layers"},
    "local_minimum": {"local", "minimum", "basin", "landscape", "stuck", "region"},
    "curvature": {"curvature", "bend", "surface", "sharp", "flat", "hessian"},
    "regularization": {"regularization", "penalty", "weight", "decay", "generalization"},
    "convergence": {"convergence", "converges", "stable", "minimum", "loss", "updates"},
}


def _updated_claims_understood(
    *,
    claims_before: dict[str, list[str]],
    concept_id: str,
    tutor_message: str,
    mastery_before: float,
    mastery_after: float,
    clarity: float,
    checkpoint_expected: bool,
    checkpoint_quality: float,
) -> dict[str, list[str]]:
    claims = {key: list(value) for key, value in claims_before.items()}
    claims.setdefault(concept_id, [])
    if mastery_after < 0.38 and (mastery_after - mastery_before) < 0.03:
        return claims

    extracted = _extract_claims_for_concept(concept_id=concept_id, tutor_message=tutor_message)
    if clarity >= 0.5 and (mastery_after - mastery_before) >= 0.04:
        for claim in extracted[:2]:
            if claim not in claims[concept_id]:
                claims[concept_id].append(claim)

    if mastery_after >= 0.55 or (checkpoint_expected and checkpoint_quality >= 0.8 and mastery_after > mastery_before):
        for claim in CANONICAL_CLAIMS.get(concept_id, [])[:2]:
            if claim not in claims[concept_id]:
                claims[concept_id].append(claim)

    claims[concept_id] = claims[concept_id][:4]
    return claims


def _extract_claims_for_concept(*, concept_id: str, tutor_message: str) -> list[str]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", tutor_message)
        if sentence.strip()
    ]
    keywords = CONCEPT_KEYWORDS.get(concept_id, set())
    matches: list[str] = []
    for sentence in sentences:
        lowered = sentence.lower()
        if "?" in sentence:
            continue
        if not any(keyword in lowered for keyword in keywords):
            continue
        if len(sentence.split()) < 6:
            continue
        cleaned = re.sub(r"\s+", " ", sentence).strip()
        if cleaned not in matches:
            matches.append(cleaned)
    return matches


def _learning_style_fit(
    *,
    learning_style: dict[str, float],
    text_features: dict[str, float],
    action_id: str,
) -> float:
    content_mix = _content_mix(text_features=text_features, action_id=action_id)
    weights = {
        "text_description": _clip01(learning_style.get("text_description", 0.5)),
        "text_examples": _clip01(learning_style.get("text_examples", 0.3)),
        "visual": _clip01(learning_style.get("visual", 0.2)),
    }
    total_weight = sum(weights.values()) or 1.0
    return _clip01(
        sum(weights[key] * content_mix[key] for key in weights) / total_weight
    )


def _content_mix(*, text_features: dict[str, float], action_id: str) -> dict[str, float]:
    text_examples = _clip01(
        0.70 * text_features["example_density"]
        + 0.20 * int(action_id in {"worked_example", "analogy"})
        + 0.10 * int(action_id == "step_by_step")
    )
    visual = _clip01(
        0.35 * text_features["structure"]
        + 0.30 * text_features["visual_structure"]
        + 0.20 * int(action_id in {"highlight_key_points", "step_by_step"})
        + 0.10 * int(any(token in action_id for token in {"analogy"}))
    )
    text_description = _clip01(
        0.55 * (1.0 - text_examples)
        + 0.20 * text_features["technical_density"]
        + 0.15 * int(action_id in {"simplify", "deepen", "summarize", "no_change"})
        + 0.10 * (1.0 - visual)
    )
    return {
        "text_description": text_description,
        "text_examples": text_examples,
        "visual": visual,
    }


def _softmax(logits: dict[str, float], temperature: float) -> dict[str, float]:
    adjusted = {key: value / max(temperature, 0.01) for key, value in logits.items()}
    max_logit = max(adjusted.values())
    exp_values = {key: math.exp(value - max_logit) for key, value in adjusted.items()}
    total = sum(exp_values.values())
    return {key: value / total for key, value in exp_values.items()}


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

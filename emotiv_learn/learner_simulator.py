from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Any

from .schemas import CANONICAL_ACTION_IDS


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def estimate_content_complexity(content_text: str) -> float:
    """Small text-complexity heuristic in [0, 1]."""
    sentences = [
        sentence.strip()
        for sentence in content_text.replace("?", ".").replace("!", ".").split(".")
        if sentence.strip()
    ]
    words = content_text.lower().split()
    if not sentences or not words:
        return 0.5

    avg_sentence_len = sum(len(sentence.split()) for sentence in sentences) / len(sentences)
    vocab_diversity = len(set(words)) / len(words)
    sentence_score = _clip01((avg_sentence_len - 8.0) / 22.0)
    vocab_score = _clip01((vocab_diversity - 0.30) / 0.60)
    return 0.6 * sentence_score + 0.4 * vocab_score


@dataclass(frozen=True)
class LearnerProfile:
    """Stable learner traits that shape which interventions work best."""

    user_id: str
    example_preference: float = 0.5
    abstraction_preference: float = 0.5
    structure_preference: float = 0.5
    challenge_preference: float = 0.5
    verbosity_tolerance: float = 0.5

    def __post_init__(self) -> None:
        for field_name, value in asdict(self).items():
            if field_name != "user_id" and not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")


@dataclass(frozen=True)
class HiddenLearnerState:
    """Ground-truth learning state owned by the simulator, not visible to System 1."""

    mastery: float
    confusion: float
    fatigue: float
    curiosity: float
    engagement: float

    def clipped(self) -> HiddenLearnerState:
        return HiddenLearnerState(
            mastery=_clip01(self.mastery),
            confusion=_clip01(self.confusion),
            fatigue=_clip01(self.fatigue),
            curiosity=_clip01(self.curiosity),
            engagement=_clip01(self.engagement),
        )


@dataclass(frozen=True)
class LearnerStep:
    user_id: str
    action_id: str
    content_complexity: float
    previous_state: HiddenLearnerState
    next_state: HiddenLearnerState
    eeg_features: list[float]
    behavioral_cues: dict[str, float | int]
    learner_response_type: str
    checkpoint_correct: int | None
    reward_signals: dict[str, float | int]


class HiddenLearnerSimulator:
    """Causal learner simulator: hidden state first, observable emissions second."""

    def __init__(
        self,
        profile: LearnerProfile,
        initial_state: HiddenLearnerState | None = None,
        seed: int | None = None,
    ) -> None:
        self.profile = profile
        self.state = (
            initial_state
            or HiddenLearnerState(
                mastery=0.25,
                confusion=0.55,
                fatigue=0.20,
                curiosity=0.45,
                engagement=0.65,
            )
        ).clipped()
        self.rng = random.Random(seed)

    def step(
        self,
        content_text: str,
        action_id: str,
        checkpoint: bool = False,
    ) -> LearnerStep:
        if action_id not in CANONICAL_ACTION_IDS:
            raise ValueError(f"unknown action_id: {action_id}")

        previous_state = self.state
        content_complexity = estimate_content_complexity(content_text)
        action_fit = self._action_fit(action_id, previous_state)
        next_state = self._transition(previous_state, action_id, action_fit, content_complexity)
        self.state = next_state

        checkpoint_correct = None
        if checkpoint:
            checkpoint_correct = int(self.rng.random() < self._checkpoint_correct_probability(next_state))

        eeg_features = self._emit_eeg_features(next_state, content_complexity)
        behavioral_cues = self._emit_behavioral_cues(next_state, content_complexity)
        learner_response_type = self._emit_response_type(next_state)
        reward_signals = self._emit_reward_signals(
            next_state=next_state,
            learner_response_type=learner_response_type,
            checkpoint_correct=checkpoint_correct,
            behavioral_cues=behavioral_cues,
        )

        return LearnerStep(
            user_id=self.profile.user_id,
            action_id=action_id,
            content_complexity=content_complexity,
            previous_state=previous_state,
            next_state=next_state,
            eeg_features=eeg_features,
            behavioral_cues=behavioral_cues,
            learner_response_type=learner_response_type,
            checkpoint_correct=checkpoint_correct,
            reward_signals=reward_signals,
        )

    def _action_fit(self, action_id: str, state: HiddenLearnerState) -> float:
        profile = self.profile
        high_support_need = 0.5 * state.confusion + 0.3 * (1.0 - state.mastery) + 0.2 * state.fatigue
        readiness_for_depth = 0.5 * state.mastery + 0.3 * state.curiosity + 0.2 * profile.challenge_preference

        fits = {
            "no_change": 0.5 * state.mastery + 0.3 * (1.0 - state.confusion) + 0.2 * state.engagement,
            "simplify": 0.7 * high_support_need + 0.3 * (1.0 - profile.challenge_preference),
            "deepen": 0.8 * readiness_for_depth + 0.2 * profile.abstraction_preference,
            "summarize": 0.5 * state.fatigue + 0.3 * (1.0 - profile.verbosity_tolerance) + 0.2 * state.confusion,
            "highlight_key_points": 0.5 * profile.structure_preference + 0.3 * state.fatigue + 0.2 * state.confusion,
            "worked_example": 0.5 * profile.example_preference + 0.3 * high_support_need + 0.2 * (1.0 - state.mastery),
            "analogy": 0.4 * state.curiosity + 0.3 * state.confusion + 0.3 * (1.0 - profile.abstraction_preference),
            "step_by_step": 0.5 * profile.structure_preference + 0.4 * high_support_need + 0.1 * (1.0 - state.mastery),
        }
        return _clip01(fits[action_id])

    def _transition(
        self,
        state: HiddenLearnerState,
        action_id: str,
        action_fit: float,
        content_complexity: float,
    ) -> HiddenLearnerState:
        overload = _clip01(0.55 * content_complexity + 0.35 * state.confusion + 0.20 * state.fatigue)
        learning_gain = (
            0.03
            + 0.16 * action_fit * state.engagement * (1.0 - state.fatigue)
            - 0.08 * overload
        )
        confusion_delta = 0.12 * overload - 0.18 * action_fit
        fatigue_delta = 0.04 + 0.08 * content_complexity - 0.04 * action_fit
        curiosity_delta = self._curiosity_delta(action_id, state, action_fit)
        engagement_delta = 0.10 * action_fit + 0.05 * state.curiosity - 0.10 * state.confusion - 0.05 * state.fatigue

        return HiddenLearnerState(
            mastery=state.mastery + learning_gain * (1.0 - state.mastery),
            confusion=state.confusion + confusion_delta,
            fatigue=state.fatigue + fatigue_delta,
            curiosity=state.curiosity + curiosity_delta,
            engagement=state.engagement + engagement_delta,
        ).clipped()

    def _curiosity_delta(self, action_id: str, state: HiddenLearnerState, action_fit: float) -> float:
        if action_id in {"deepen", "analogy"}:
            return 0.08 * action_fit - 0.04 * state.confusion
        if action_id in {"simplify", "step_by_step"}:
            return 0.03 * action_fit - 0.03 * self.profile.challenge_preference
        if action_id == "no_change":
            return -0.03 * (1.0 - state.mastery)
        return 0.02 * action_fit

    def _checkpoint_correct_probability(self, state: HiddenLearnerState) -> float:
        probability = 0.10 + 0.80 * state.mastery - 0.20 * state.confusion - 0.10 * state.fatigue
        return _clip01(probability)

    def _emit_eeg_features(self, state: HiddenLearnerState, content_complexity: float) -> list[float]:
        """Emit a synthetic 62-dim EEG-like vector conditioned on hidden load."""
        load = self._hidden_load(state, content_complexity)
        features = [0.0] * 62
        for index in range(56):
            band_index = index // 14
            base = [0.28, 0.32, 0.25, 0.15][band_index]
            load_effect = [0.10, -0.08, 0.04, 0.02][band_index] * load
            noise = self.rng.gauss(0.0, 0.015)
            features[index] = _clip01(base + load_effect + noise)

        features[56] = self.rng.gauss(0.0, 0.03)
        features[57] = self.rng.gauss(0.0, 0.05)
        for index in range(58, 62):
            features[index] = max(0.0, 0.45 + 1.6 * load + self.rng.gauss(0.0, 0.10))
        return features

    def _emit_behavioral_cues(
        self,
        state: HiddenLearnerState,
        content_complexity: float,
    ) -> dict[str, float | int]:
        load = self._hidden_load(state, content_complexity)
        time_on_chunk = 35.0 + 85.0 * load + 35.0 * content_complexity + self.rng.gauss(0.0, 5.0)
        scroll_rate = _clip01(1.0 - 0.55 * load - 0.25 * content_complexity + self.rng.gauss(0.0, 0.04))
        reread_lambda = max(0.05, 0.5 + 2.5 * state.confusion + 1.0 * content_complexity)
        reread_count = min(6, self._sample_poisson(reread_lambda))
        return {
            "time_on_chunk": round(max(20.0, time_on_chunk), 1),
            "scroll_rate": round(scroll_rate, 3),
            "reread_count": reread_count,
        }

    def _emit_response_type(self, state: HiddenLearnerState) -> str:
        logits = {
            "clarify": -0.8 + 2.4 * state.confusion + 0.8 * state.fatigue - 0.8 * state.mastery,
            "branch": -1.2 + 1.6 * state.curiosity + 0.8 * state.mastery - 0.6 * state.confusion,
            "continue": 0.4 + 1.8 * state.mastery + 0.6 * state.engagement - 1.4 * state.confusion,
        }
        return self._sample_categorical(logits)

    def _emit_reward_signals(
        self,
        next_state: HiddenLearnerState,
        learner_response_type: str,
        checkpoint_correct: int | None,
        behavioral_cues: dict[str, float | int],
    ) -> dict[str, float | int]:
        return {
            "checkpoint_occurred": int(checkpoint_correct is not None),
            "checkpoint_correct": int(checkpoint_correct or 0),
            "progress_signal": round(next_state.mastery, 3),
            "deeper_request": int(learner_response_type == "branch"),
            "continue_signal": int(learner_response_type == "continue"),
            "clarify_signal": int(learner_response_type == "clarify"),
            "repeated_clarify_same_concept": int(
                learner_response_type == "clarify" and behavioral_cues["reread_count"] >= 2
            ),
            "hesitation_signal": int(float(behavioral_cues["time_on_chunk"]) > 100.0),
            "abandonment": int(next_state.fatigue > 0.95 and next_state.engagement < 0.20),
        }

    @staticmethod
    def _hidden_load(state: HiddenLearnerState, content_complexity: float) -> float:
        return _clip01(
            0.40 * state.confusion
            + 0.25 * state.fatigue
            + 0.20 * content_complexity
            + 0.15 * (1.0 - state.mastery)
        )

    def _sample_categorical(self, logits: dict[str, float]) -> str:
        max_logit = max(logits.values())
        exp_values = {key: math.exp(value - max_logit) for key, value in logits.items()}
        total = sum(exp_values.values())
        draw = self.rng.random()
        cumulative = 0.0
        for key, value in exp_values.items():
            cumulative += value / total
            if draw <= cumulative:
                return key
        return next(reversed(exp_values))

    def _sample_poisson(self, lam: float) -> int:
        threshold = math.exp(-lam)
        count = 0
        product = 1.0
        while product > threshold:
            count += 1
            product *= self.rng.random()
        return count - 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "profile": asdict(self.profile),
            "state": asdict(self.state),
        }

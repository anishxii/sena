from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
import re
from typing import Any

from .knowledge_scenarios import TopicScenario


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


ADVANCE_MASTERY_THRESHOLD = 0.62
MAX_STEPS_PER_CONCEPT = 3
MIN_FORCED_ADVANCE_MASTERY = 0.45


@dataclass(frozen=True)
class KnowledgeAgentProfile:
    user_id: str
    initial_knowledge_level: float = 0.35
    curiosity: float = 0.60
    confidence: float = 0.45
    fatigue: float = 0.15
    engagement: float = 0.70
    attention: float = 0.72
    preferred_style: dict[str, float] | None = None


@dataclass(frozen=True)
class KnowledgeAgentState:
    knowledge_base: tuple[str, ...]
    concept_mastery: dict[str, float]
    confusion: float
    confidence: float
    curiosity: float
    fatigue: float
    engagement: float
    attention: float
    current_concept_index: int
    current_concept_steps: int
    steps_taken: int


@dataclass(frozen=True)
class KnowledgeTurn:
    concept_id: str
    tutor_message: str
    action_id: str
    checkpoint_expected: bool
    state_before: KnowledgeAgentState
    state_after: KnowledgeAgentState
    response_type: str
    reprompt: str | None
    self_reported_confidence: float
    checkpoint_answer: str | None
    checkpoint_correct: bool | None
    progress_signal: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "tutor_message": self.tutor_message,
            "action_id": self.action_id,
            "checkpoint_expected": self.checkpoint_expected,
            "state_before": asdict(self.state_before),
            "state_after": asdict(self.state_after),
            "response_type": self.response_type,
            "reprompt": self.reprompt,
            "self_reported_confidence": self.self_reported_confidence,
            "checkpoint_answer": self.checkpoint_answer,
            "checkpoint_correct": self.checkpoint_correct,
            "progress_signal": self.progress_signal,
        }


class KnowledgeAgent:
    def __init__(self, scenario: TopicScenario, profile: KnowledgeAgentProfile, seed: int = 0) -> None:
        self.scenario = scenario
        self.profile = profile
        self.rng = random.Random(seed)
        self.seed = seed
        self.base_learning_rate = self.rng.uniform(0.88, 1.12)
        self.step_variability = self.rng.uniform(0.06, 0.16)
        self.knowledge_transfer_bias = self.rng.uniform(0.85, 1.15)
        self.reprompt_detail_bias = self.rng.uniform(0.9, 1.1)
        self.state = self._initial_state()

    def _initial_state(self) -> KnowledgeAgentState:
        initial_mastery = {concept_id: 0.10 for concept_id in self.scenario.concept_ids}
        initial_mastery["neural_networks"] = max(self.profile.initial_knowledge_level, 0.45)
        return KnowledgeAgentState(
            knowledge_base=tuple(self.scenario.initial_knowledge_base),
            concept_mastery=initial_mastery,
            confusion=0.55,
            confidence=self.profile.confidence,
            curiosity=self.profile.curiosity,
            fatigue=self.profile.fatigue,
            engagement=self.profile.engagement,
            attention=self.profile.attention,
            current_concept_index=0,
            current_concept_steps=0,
            steps_taken=0,
        )

    def current_concept_id(self) -> str:
        return self.scenario.ordered_concepts[self.state.current_concept_index].concept_id

    def consume_tutor_step(
        self,
        *,
        concept_id: str,
        tutor_message: str,
        action_id: str,
        checkpoint_expected: bool,
    ) -> KnowledgeTurn:
        before = self.state
        evaluation = _evaluate_tutor_message(tutor_message=tutor_message, action_id=action_id)
        content_context = self._content_context(tutor_message=tutor_message, concept_id=concept_id)
        after = self._next_state(
            before=before,
            concept_id=concept_id,
            evaluation=evaluation,
            content_context=content_context,
            checkpoint_expected=checkpoint_expected,
        )
        response_type = self._response_type(after, concept_id)
        reprompt = self._reprompt(after, concept_id, response_type, content_context)
        checkpoint_answer = self._checkpoint_answer(after, concept_id, content_context) if checkpoint_expected else None
        checkpoint_correct = None
        if checkpoint_expected:
            probability = _clip01(0.18 + 0.72 * after.concept_mastery[concept_id] - 0.22 * after.confusion)
            checkpoint_correct = self.rng.random() < probability

        turn = KnowledgeTurn(
            concept_id=concept_id,
            tutor_message=tutor_message,
            action_id=action_id,
            checkpoint_expected=checkpoint_expected,
            state_before=before,
            state_after=after,
            response_type=response_type,
            reprompt=reprompt,
            self_reported_confidence=round(after.confidence, 4),
            checkpoint_answer=checkpoint_answer,
            checkpoint_correct=checkpoint_correct,
            progress_signal=round(after.concept_mastery[concept_id] - before.concept_mastery.get(concept_id, 0.0), 4),
        )
        self.state = after
        return turn

    def advance_if_ready(self, concept_id: str) -> None:
        mastery = self.state.concept_mastery.get(concept_id, 0.0)
        should_advance = mastery >= ADVANCE_MASTERY_THRESHOLD or (
            self.state.current_concept_steps >= MAX_STEPS_PER_CONCEPT
            and mastery >= MIN_FORCED_ADVANCE_MASTERY
        )
        if not should_advance:
            return
        if self.state.current_concept_index >= len(self.scenario.ordered_concepts) - 1:
            return
        self.state = KnowledgeAgentState(
            knowledge_base=self.state.knowledge_base,
            concept_mastery=dict(self.state.concept_mastery),
            confusion=max(0.20, self.state.confusion - 0.10),
            confidence=self.state.confidence,
            curiosity=self.state.curiosity,
            fatigue=self.state.fatigue,
            engagement=self.state.engagement,
            attention=self.state.attention,
            current_concept_index=self.state.current_concept_index + 1,
            current_concept_steps=0,
            steps_taken=self.state.steps_taken,
        )

    def _next_state(
        self,
        *,
        before: KnowledgeAgentState,
        concept_id: str,
        evaluation: dict[str, float],
        content_context: dict[str, Any],
        checkpoint_expected: bool,
    ) -> KnowledgeAgentState:
        mastery_before = before.concept_mastery.get(concept_id, 0.0)
        prereq_bonus = self._prerequisite_bonus(before, concept_id)
        knowledge_support = self._knowledge_base_support(before, concept_id)
        step_factor = max(0.75, min(1.25, 1.0 + self.rng.gauss(0.0, self.step_variability)))
        content_alignment = float(content_context["content_alignment"])
        evidence_strength = float(content_context["evidence_strength"])
        effective_instruction = (
            0.24 * evaluation["clarity"]
            + 0.16 * evaluation["structure"]
            + 0.14 * evaluation["supportiveness"]
            + 0.08 * evaluation["novelty"]
            + 0.18 * prereq_bonus
            + 0.12 * knowledge_support
            + 0.08 * content_alignment
        )
        learning_gain = (
            0.15
            * self.base_learning_rate
            * self.knowledge_transfer_bias
            * step_factor
            * effective_instruction
            * (1.0 - mastery_before)
        )
        learning_gain -= 0.06 * evaluation["overload"]
        learning_gain += 0.05 * evidence_strength
        if checkpoint_expected:
            learning_gain += 0.03 * evaluation["checkpoint_ready"]
        mastery_after = _clip01(mastery_before + learning_gain)

        concept_mastery = dict(before.concept_mastery)
        concept_mastery[concept_id] = mastery_after

        knowledge_base = list(before.knowledge_base)
        concept = self.scenario.concept(concept_id)
        if mastery_after >= 0.62 and concept.canonical_claim not in knowledge_base:
            knowledge_base.append(concept.canonical_claim)
        for inferred_claim in content_context["candidate_claims"]:
            if inferred_claim not in knowledge_base and mastery_after >= 0.50:
                knowledge_base.append(inferred_claim)

        confusion = _clip01(0.60 * (1.0 - mastery_after) + 0.35 * evaluation["overload"] - 0.22 * before.confidence)
        confidence = _clip01(before.confidence + 0.14 * (mastery_after - mastery_before) + 0.05 * evaluation["clarity"] - 0.08 * evaluation["overload"])
        curiosity = _clip01(before.curiosity + 0.06 * evaluation["novelty"] - 0.03 * int(mastery_after > 0.85))
        fatigue = _clip01(before.fatigue + 0.04 * evaluation["overload"] - 0.02 * evaluation["supportiveness"])
        engagement = _clip01(before.engagement + 0.06 * evaluation["supportiveness"] + 0.04 * evaluation["novelty"] - 0.05 * fatigue)
        attention = _clip01(before.attention + 0.04 * evaluation["structure"] - 0.05 * evaluation["overload"])

        return KnowledgeAgentState(
            knowledge_base=tuple(knowledge_base),
            concept_mastery=concept_mastery,
            confusion=confusion,
            confidence=confidence,
            curiosity=curiosity,
            fatigue=fatigue,
            engagement=engagement,
            attention=attention,
            current_concept_index=before.current_concept_index,
            current_concept_steps=before.current_concept_steps + 1,
            steps_taken=before.steps_taken + 1,
        )

    def _prerequisite_bonus(self, state: KnowledgeAgentState, concept_id: str) -> float:
        prerequisites = self.scenario.concept(concept_id).prerequisites
        if not prerequisites:
            return 1.0
        return sum(state.concept_mastery.get(concept, 0.0) for concept in prerequisites) / len(prerequisites)

    def _knowledge_base_support(self, state: KnowledgeAgentState, concept_id: str) -> float:
        concept = self.scenario.concept(concept_id)
        prerequisites = concept.prerequisites
        knowledge_base = {claim.lower() for claim in state.knowledge_base}

        if prerequisites:
            prereq_claim_support = sum(
                1
                for prereq_id in prerequisites
                if self.scenario.concept(prereq_id).canonical_claim.lower() in knowledge_base
            ) / len(prerequisites)
            prereq_mastery_support = sum(state.concept_mastery.get(prereq_id, 0.0) for prereq_id in prerequisites) / len(prerequisites)
        else:
            prereq_claim_support = 1.0
            prereq_mastery_support = 1.0

        known_concept_support = sum(1 for mastery in state.concept_mastery.values() if mastery >= 0.45) / max(
            len(state.concept_mastery),
            1,
        )
        return _clip01(
            0.45 * prereq_mastery_support
            + 0.35 * prereq_claim_support
            + 0.20 * known_concept_support
        )

    def _content_context(self, *, tutor_message: str, concept_id: str) -> dict[str, Any]:
        concept = self.scenario.concept(concept_id)
        keywords = {
            concept.label.lower(),
            *[token for token in re.findall(r"[A-Za-z']+", concept.canonical_claim.lower()) if len(token) > 3],
        }
        for prerequisite_id in concept.prerequisites:
            prerequisite = self.scenario.concept(prerequisite_id)
            keywords.add(prerequisite.label.lower())
            keywords.update(
                token
                for token in re.findall(r"[A-Za-z']+", prerequisite.canonical_claim.lower())
                if len(token) > 3
            )

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", tutor_message.strip())
            if sentence.strip()
        ]
        relevant_sentences = [
            sentence
            for sentence in sentences
            if any(keyword in sentence.lower() for keyword in keywords)
        ]
        if not relevant_sentences and sentences:
            relevant_sentences = sentences[:2]

        candidate_claims = [
            self._normalize_claim_text(sentence)
            for sentence in relevant_sentences[:2]
            if "?" not in sentence
        ]
        evidence_sentences = [sentence for sentence in relevant_sentences if "?" not in sentence] or relevant_sentences
        content_alignment = _clip01(len(relevant_sentences) / max(len(sentences), 1))
        evidence_strength = _clip01(
            0.45 * content_alignment
            + 0.30 * min(sum(len(sentence.split()) for sentence in evidence_sentences) / 36.0, 1.0)
            + 0.25 * int("?" in tutor_message or "because" in tutor_message.lower() or "so that" in tutor_message.lower())
        )
        return {
            "relevant_sentences": evidence_sentences,
            "candidate_claims": candidate_claims,
            "content_alignment": content_alignment,
            "evidence_strength": evidence_strength,
        }

    def _normalize_claim_text(self, sentence: str) -> str:
        cleaned = re.sub(r"\s+", " ", sentence.strip())
        if not cleaned:
            return cleaned
        cleaned = cleaned.rstrip(".!?")
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        return f"{cleaned}."

    def _response_type(self, state: KnowledgeAgentState, concept_id: str) -> str:
        mastery = state.concept_mastery.get(concept_id, 0.0)
        if state.confusion >= 0.58 or mastery < 0.42:
            return "clarify"
        if state.curiosity >= 0.68 and mastery >= 0.58:
            return "branch"
        return "continue"

    def _reprompt(self, state: KnowledgeAgentState, concept_id: str, response_type: str, content_context: dict[str, Any]) -> str | None:
        concept = self.scenario.concept(concept_id)
        known_labels = [
            self.scenario.concept(cid).label
            for cid, mastery in state.concept_mastery.items()
            if mastery >= 0.45 and cid in self.scenario.concept_ids
        ]
        known_claims = list(state.knowledge_base)[-2:]
        relevant_sentences = content_context["relevant_sentences"]
        recent_evidence = relevant_sentences[0] if relevant_sentences else concept.canonical_claim
        anchor = known_labels[-1] if known_labels else concept.label
        confidence_text = "I feel mostly grounded in it" if state.confidence >= 0.6 else "I only partially understand it"
        if response_type == "continue":
            if known_claims:
                return (
                    f"I think this is starting to click. Based on what I already know about {anchor}, "
                    f"I can connect this step to the idea that {known_claims[-1].lower()} "
                    f"and to your point that {recent_evidence.lower()} Please continue to the next part."
                )
            return "I think I understand this part. Let's keep going."
        if response_type == "clarify":
            anchor = known_labels[0] if known_labels else concept.label
            if known_claims:
                return (
                    f"I understand the part about {anchor}, and {confidence_text}, but I am still confused about "
                    f"how {concept.label} fits into the full process. Can you explain it again more concretely "
                    f"and relate it to {known_claims[-1].lower()}? I especially need help with the part where you said "
                    f"'{recent_evidence}'."
                )
            return (
                f"I understand some of the setup around {anchor}, but I am still confused about how "
                f"{concept.label} actually works. Can you break it down more explicitly using the part where you said "
                f"'{recent_evidence}'?"
            )
        if known_claims:
            return (
                f"I can see how {anchor} connects to what I already know, especially that {known_claims[-1].lower()} "
                f"You also mentioned that {recent_evidence.lower()} Can we go one level deeper and understand the mechanism behind {concept.label}?"
            )
        return f"I see how {anchor} connects here. Can we go one level deeper on {concept.label}?"

    def _checkpoint_answer(self, state: KnowledgeAgentState, concept_id: str, content_context: dict[str, Any]) -> str:
        mastery = state.concept_mastery.get(concept_id, 0.0)
        concept = self.scenario.concept(concept_id)
        recent_evidence = content_context["relevant_sentences"][0] if content_context["relevant_sentences"] else concept.canonical_claim
        if mastery >= 0.72:
            return f"{concept.canonical_claim} In this step, the key idea was that {recent_evidence.lower()}"
        if mastery >= 0.45:
            return (
                f"I think {concept.label} is related to {recent_evidence.lower()}, "
                "but I may still be missing some of the mechanism."
            )
        return (
            f"I'm not fully sure yet how {concept.label} works. "
            f"The part I caught was that {recent_evidence.lower()}"
        )


def _evaluate_tutor_message(tutor_message: str, action_id: str) -> dict[str, float]:
    words = re.findall(r"[A-Za-z']+", tutor_message.lower())
    word_count = len(words)
    sentences = max(len(re.findall(r"[.!?]", tutor_message)), 1)
    avg_sentence_len = word_count / sentences
    unique_ratio = len(set(words)) / max(word_count, 1)
    bullet_count = len(re.findall(r"(^|\n)\s*(-|\*|\d+\.)\s+", tutor_message))
    question_count = tutor_message.count("?")
    technical_terms = sum(
        1
        for word in words
        if word in {"gradient", "loss", "parameter", "chain", "backpropagation", "derivative", "convergence", "weight"}
    )
    example_terms = sum(1 for word in words if word in {"example", "imagine", "suppose", "step"})
    supportive_terms = sum(1 for word in words if word in {"notice", "remember", "help", "think", "consider"})

    complexity = _clip01(0.35 * (avg_sentence_len / 26.0) + 0.35 * unique_ratio + 0.30 * (technical_terms / 10.0))
    structure = _clip01(0.25 + 0.15 * min(bullet_count, 4) + 0.10 * min(example_terms, 4))
    supportiveness = _clip01(0.35 + 0.20 * min(supportive_terms, 4) / 4.0 + 0.10 * structure)
    novelty = _clip01(0.25 + 0.25 * min(example_terms, 4) / 4.0 + 0.20 * unique_ratio + 0.10 * int(action_id in {"deepen", "analogy"}))
    overload = _clip01(0.55 * complexity - 0.20 * structure)
    checkpoint_ready = 1.0 if question_count else 0.5

    return {
        "complexity": complexity,
        "structure": structure,
        "supportiveness": supportiveness,
        "novelty": novelty,
        "overload": overload,
        "clarity": _clip01(0.40 + 0.30 * structure + 0.15 * supportiveness - 0.20 * overload),
        "checkpoint_ready": checkpoint_ready,
    }

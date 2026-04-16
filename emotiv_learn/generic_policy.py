from __future__ import annotations

from dataclasses import dataclass

from .knowledge_agent import KnowledgeAgentState
from .knowledge_scenarios import TopicScenario
from .openai_client import OpenAIChatClient


GENERIC_TUTOR_SYSTEM_PROMPT = """You are a clear tutor teaching machine learning.

Respond directly to the learner's latest message and help them understand the current concept.
Keep the lesson focused, practical, and supportive.
If a checkpoint is requested, end with one short question.
Return only the tutor message."""


@dataclass(frozen=True)
class GenericPolicyInput:
    scenario: TopicScenario
    concept_id: str
    learner_state: KnowledgeAgentState
    learner_message: str | None
    checkpoint_expected: bool


class GenericTutorPolicy:
    def __init__(self, client: OpenAIChatClient) -> None:
        self.client = client

    def next_tutor_step(self, policy_input: GenericPolicyInput) -> str:
        concept = policy_input.scenario.concept(policy_input.concept_id)
        knowledge_preview = list(policy_input.learner_state.knowledge_base)[-3:]
        user_prompt = f"""Topic: {policy_input.scenario.title}

Current concept: {concept.label}
Current learner message: {policy_input.learner_message or "The learner is ready for the next explanation."}
Learner already knows:
{knowledge_preview}

Current estimated mastery for this concept: {policy_input.learner_state.concept_mastery.get(policy_input.concept_id, 0.0):.2f}
Checkpoint required: {policy_input.checkpoint_expected}

Teach the concept in a standard, non-personalized way."""
        return self.client.complete_text(
            [
                {"role": "system", "content": GENERIC_TUTOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=700,
            temperature=0.35,
        )

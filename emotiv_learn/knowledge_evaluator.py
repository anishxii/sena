from __future__ import annotations

from dataclasses import dataclass

from .knowledge_agent import KnowledgeAgentState
from .knowledge_scenarios import TopicScenario


@dataclass(frozen=True)
class KnowledgeEvaluation:
    goal_coverage_score: float
    knowledge_quality_score: float
    missing_goal_claims: list[str]
    achieved_goal_claims: list[str]
    goal_reached: bool


def evaluate_knowledge_state(scenario: TopicScenario, state: KnowledgeAgentState) -> KnowledgeEvaluation:
    knowledge_base = set(state.knowledge_base)
    achieved = [claim for claim in scenario.goal_knowledge_base if claim in knowledge_base]
    missing = [claim for claim in scenario.goal_knowledge_base if claim not in knowledge_base]
    goal_concepts = [concept.concept_id for concept in scenario.ordered_concepts if concept.canonical_claim in scenario.goal_knowledge_base]
    knowledge_quality = sum(state.concept_mastery.get(concept_id, 0.0) for concept_id in goal_concepts) / max(len(goal_concepts), 1)
    goal_coverage = len(achieved) / max(len(scenario.goal_knowledge_base), 1)
    goal_reached = goal_coverage >= 1.0 and all(state.concept_mastery.get(concept_id, 0.0) >= 0.72 for concept_id in goal_concepts)
    return KnowledgeEvaluation(
        goal_coverage_score=goal_coverage,
        knowledge_quality_score=knowledge_quality,
        missing_goal_claims=missing,
        achieved_goal_claims=achieved,
        goal_reached=goal_reached,
    )

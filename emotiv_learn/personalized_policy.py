from __future__ import annotations

from dataclasses import dataclass

from .decision_engine import DecisionEngine
from .eeg import EEGObservationContext, EEGProvider, EEGWindow, estimate_time_on_chunk
from .knowledge_agent import KnowledgeAgentState
from .knowledge_scenarios import TopicScenario
from .llm_contracts import TutorPromptInput, build_tutor_messages, compute_reward_from_interpreted
from .live_training import LIVE_FEATURE_NAMES, LiveLLMStateBuilder, LiveStateInput
from .openai_client import OpenAIChatClient
from .reprompt_analyzer import analyze_reprompt
from .schemas import ACTION_BANK, Outcome, RewardEvent, SemanticSignals, TaskResult


@dataclass(frozen=True)
class PersonalizedPolicyTurn:
    action_id: str
    tutor_message: str
    interpreted: dict
    eeg_window: EEGWindow
    reward: float


class PersonalizedTutorPolicy:
    def __init__(self, client: OpenAIChatClient, eeg_provider: EEGProvider, seed: int = 0) -> None:
        self.client = client
        self.eeg_provider = eeg_provider
        self.state_builder = LiveLLMStateBuilder()
        self.engine = DecisionEngine(
            feature_dim=len(LIVE_FEATURE_NAMES),
            epsilon=0.10,
            use_personalization=True,
            seed=seed,
            reward_clip_abs=1.5,
            update_clip_abs=0.2,
            l2_weight_decay=0.001,
        )
        self.previous_interpreted: dict | None = None
        self.previous_student_response: dict | None = None
        self.previous_reward: float = 0.0
        self.previous_eeg_window: EEGWindow | None = None

    def next_action_and_message(
        self,
        *,
        timestamp: int,
        user_id: str,
        scenario: TopicScenario,
        concept_id: str,
        learner_state: KnowledgeAgentState,
        learner_message: str | None,
        checkpoint_expected: bool,
    ) -> tuple[str, str]:
        state = self.state_builder.build_state(
            LiveStateInput(
                timestamp=timestamp,
                user_id=user_id,
                topic_id=scenario.topic_id,
                task_type="learn",
                difficulty="medium",
                turn_index=timestamp,
                max_turns=max(10, timestamp),
                interpreted=self.previous_interpreted,
                student_response=self.previous_student_response,
                previous_reward=self.previous_reward,
                eeg_window=self.previous_eeg_window,
            )
        )
        action_id = self.engine.select_action(self.engine.score_actions(state, ACTION_BANK)).action_id
        messages = build_tutor_messages(
            TutorPromptInput(
                topic=scenario.title,
                concept_id=concept_id,
                conversation_summary=learner_message or "The learner is ready for the next concept step.",
                load_level=_load_level(self.previous_interpreted),
                behavior_summary=_behavior_summary(self.previous_interpreted),
                last_followup_type=(self.previous_interpreted or {}).get("followup_type", "unknown"),
                action_id=action_id,
                length_target="short",
                difficulty_target="medium",
                include_checkpoint=checkpoint_expected,
            )
        )
        tutor_message = self.client.complete_text(messages, max_tokens=700, temperature=0.35)
        return action_id, tutor_message

    def observe_turn(
        self,
        *,
        timestamp: int,
        user_id: str,
        scenario: TopicScenario,
        concept_id: str,
        action_id: str,
        tutor_message: str,
        knowledge_turn,
    ) -> PersonalizedPolicyTurn:
        analysis = analyze_reprompt(
            reprompt=knowledge_turn.reprompt,
            response_type=knowledge_turn.response_type,
            self_reported_confidence=knowledge_turn.self_reported_confidence,
            current_mastery=knowledge_turn.state_after.concept_mastery.get(concept_id, 0.0),
        )
        interpreted = analysis.to_interpreted()
        time_on_chunk = estimate_time_on_chunk(tutor_message)
        eeg_window = self.eeg_provider.observe(
            EEGObservationContext(
                timestamp=timestamp,
                user_id=user_id,
                concept_id=concept_id,
                action_id=action_id,
                tutor_message=tutor_message,
                time_on_chunk=time_on_chunk,
                hidden_state={
                    "concept_mastery": knowledge_turn.state_after.concept_mastery,
                    "fatigue": knowledge_turn.state_after.fatigue,
                    "attention": knowledge_turn.state_after.attention,
                    "confidence": knowledge_turn.state_after.confidence,
                    "engagement": knowledge_turn.state_after.engagement,
                },
                observable_signals={
                    **interpreted,
                    "confidence": knowledge_turn.state_after.confidence,
                    "attention": knowledge_turn.state_after.attention,
                    "fatigue": knowledge_turn.state_after.fatigue,
                },
            )
        )
        reward = compute_reward_from_interpreted(interpreted)
        state = self.state_builder.build_state(
            LiveStateInput(
                timestamp=timestamp,
                user_id=user_id,
                topic_id=scenario.topic_id,
                task_type="learn",
                difficulty="medium",
                turn_index=timestamp,
                max_turns=max(10, timestamp),
                interpreted=self.previous_interpreted,
                student_response=self.previous_student_response,
                previous_reward=self.previous_reward,
                eeg_window=self.previous_eeg_window,
            )
        )
        self.engine.update(
            RewardEvent(
                timestamp=timestamp,
                user_id=user_id,
                state_features=state.features,
                action_id=action_id,
                reward=reward,
                outcome=Outcome(
                    timestamp=timestamp,
                    user_id=user_id,
                    action_id=action_id,
                    task_result=TaskResult(
                        correct=int(bool(knowledge_turn.checkpoint_correct)) if knowledge_turn.checkpoint_correct is not None else None,
                        latency_s=time_on_chunk,
                        reread=None,
                        completed=1,
                        abandoned=0,
                    ),
                    semantic_signals=SemanticSignals(
                        followup_text=knowledge_turn.reprompt,
                        followup_type=knowledge_turn.response_type,
                        confusion_score=float(interpreted["confusion_score"]),
                        comprehension_score=float(interpreted["comprehension_score"]),
                        engagement_score=float(interpreted["engagement_score"]),
                        pace_fast_score=float(interpreted["pace_fast_score"]),
                        pace_slow_score=float(interpreted["pace_slow_score"]),
                    ),
                    raw={
                        "concept_id": concept_id,
                        "checkpoint_expected": knowledge_turn.checkpoint_expected,
                        "checkpoint_correct": knowledge_turn.checkpoint_correct,
                    },
                ),
            )
        )
        self.previous_interpreted = interpreted
        self.previous_student_response = {
            "response_type": knowledge_turn.response_type,
            "self_reported_confidence": knowledge_turn.self_reported_confidence,
        }
        self.previous_reward = reward
        self.previous_eeg_window = eeg_window
        return PersonalizedPolicyTurn(
            action_id=action_id,
            tutor_message=tutor_message,
            interpreted=interpreted,
            eeg_window=eeg_window,
            reward=reward,
        )


def _load_level(interpreted: dict | None) -> str:
    if interpreted is None:
        return "medium"
    confusion = float(interpreted.get("confusion_score", 0.5))
    if confusion >= 0.65:
        return "high"
    if confusion <= 0.30:
        return "low"
    return "medium"


def _behavior_summary(interpreted: dict | None) -> str:
    if interpreted is None:
        return "no prior learner response"
    return (
        f"confusion={interpreted.get('confusion_score', 0.5):.2f}, "
        f"comprehension={interpreted.get('comprehension_score', 0.5):.2f}, "
        f"engagement={interpreted.get('engagement_score', 0.5):.2f}"
    )

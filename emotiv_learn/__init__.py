"""Legacy package exports for backward compatibility during the refactor."""

from .decision_engine import DecisionEngine
from .eeg import (
    EEGObservationContext,
    EEGProvider,
    EEGProxyState,
    EEGWindow,
    HeuristicTargetEEGMapper,
    SyntheticEEGProvider,
    build_eeg_provider,
)
from .llm_contracts import compute_reward_from_interpreted
from systems.system1_decision.schemas import ACTION_BANK
from .student_model import HiddenKnowledgeState, HiddenKnowledgeStudent, KnowledgeState, NeuroState, default_hidden_knowledge_state

__all__ = [
    "ACTION_BANK",
    "DecisionEngine",
    "EEGObservationContext",
    "EEGProvider",
    "EEGProxyState",
    "EEGWindow",
    "HeuristicTargetEEGMapper",
    "HiddenKnowledgeState",
    "HiddenKnowledgeStudent",
    "KnowledgeState",
    "NeuroState",
    "SyntheticEEGProvider",
    "build_eeg_provider",
    "compute_reward_from_interpreted",
    "default_hidden_knowledge_state",
]

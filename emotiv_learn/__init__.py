"""Core package for the Emotiv Learn decision policy system."""

from .cog_bci_metadata import NBackRecordingSummary, build_subject_nback_recording_summaries
from .cog_bci_proxy_model import COGBCIProxyRegressor, fit_cog_bci_proxy_regressor
from .decision_engine import DecisionEngine
from .eeg import (
    EEGObservationContext,
    EEGProvider,
    EEGProxyState,
    EEGWindow,
    HeuristicTargetEEGMapper,
    RetrievedEEGProvider,
    SyntheticEEGProvider,
    build_eeg_provider,
)
from .eeg_mapper import STEWWorkloadFeatureMapper, fit_stew_workload_feature_mapper
from .knowledge_agent import KnowledgeAgent, KnowledgeAgentProfile, KnowledgeAgentState, KnowledgeTurn
from .knowledge_evaluator import KnowledgeEvaluation, evaluate_knowledge_state
from .knowledge_scenarios import BACKPROP_SCENARIO, ConceptSpec, TopicScenario
from .llm_contracts import compute_reward_from_interpreted
from .schemas import ACTION_BANK
from .student_model import HiddenKnowledgeState, HiddenKnowledgeStudent, KnowledgeState, NeuroState, default_hidden_knowledge_state

__all__ = [
    "ACTION_BANK",
    "COGBCIProxyRegressor",
    "NBackRecordingSummary",
    "DecisionEngine",
    "EEGObservationContext",
    "EEGProvider",
    "EEGProxyState",
    "EEGWindow",
    "STEWWorkloadFeatureMapper",
    "HeuristicTargetEEGMapper",
    "HiddenKnowledgeState",
    "HiddenKnowledgeStudent",
    "KnowledgeState",
    "NeuroState",
    "KnowledgeAgent",
    "KnowledgeAgentProfile",
    "KnowledgeAgentState",
    "KnowledgeEvaluation",
    "KnowledgeTurn",
    "RetrievedEEGProvider",
    "SyntheticEEGProvider",
    "BACKPROP_SCENARIO",
    "build_subject_nback_recording_summaries",
    "build_eeg_provider",
    "fit_cog_bci_proxy_regressor",
    "fit_stew_workload_feature_mapper",
    "ConceptSpec",
    "TopicScenario",
    "compute_reward_from_interpreted",
    "default_hidden_knowledge_state",
    "evaluate_knowledge_state",
]

"""Core package for the Emotiv Learn decision policy system."""

from .decision_engine import DecisionEngine
from .eeg import (
    DEFAULT_USER_TO_STEW_SUBJECT,
    EEGObservationContext,
    EEGProvider,
    EEGWindow,
    MatchedRealEEGProvider,
    SyntheticEEGProvider,
    build_eeg_provider,
)
from .eeg_features import EEG_FEATURE_NAMES
from .generic_policy import GenericPolicyInput, GenericTutorPolicy
from .knowledge_agent import KnowledgeAgent, KnowledgeAgentProfile, KnowledgeAgentState, KnowledgeTurn
from .knowledge_evaluator import KnowledgeEvaluation, evaluate_knowledge_state
from .knowledge_scenarios import BACKPROP_SCENARIO, ConceptSpec, TopicScenario
from .learner_simulator import HiddenLearnerSimulator, HiddenLearnerState, LearnerProfile
from .llm_contracts import compute_reward_from_interpreted
from .personalized_policy import PersonalizedPolicyTurn, PersonalizedTutorPolicy
from .reprompt_analyzer import RepromptAnalysis, analyze_reprompt
from .schemas import ACTION_BANK
from .stew_index import STEWFeatureIndex, build_stew_feature_index, load_feature_index, save_feature_index

__all__ = [
    "ACTION_BANK",
    "analyze_reprompt",
    "BACKPROP_SCENARIO",
    "build_eeg_provider",
    "build_stew_feature_index",
    "ConceptSpec",
    "DEFAULT_USER_TO_STEW_SUBJECT",
    "DecisionEngine",
    "EEG_FEATURE_NAMES",
    "EEGObservationContext",
    "EEGProvider",
    "EEGWindow",
    "evaluate_knowledge_state",
    "GenericPolicyInput",
    "GenericTutorPolicy",
    "HiddenLearnerSimulator",
    "HiddenLearnerState",
    "KnowledgeAgent",
    "KnowledgeAgentProfile",
    "KnowledgeAgentState",
    "KnowledgeEvaluation",
    "KnowledgeTurn",
    "LearnerProfile",
    "load_feature_index",
    "MatchedRealEEGProvider",
    "PersonalizedPolicyTurn",
    "PersonalizedTutorPolicy",
    "RepromptAnalysis",
    "save_feature_index",
    "SyntheticEEGProvider",
    "STEWFeatureIndex",
    "compute_reward_from_interpreted",
    "TopicScenario",
]

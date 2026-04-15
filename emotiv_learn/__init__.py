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
from .learner_simulator import HiddenLearnerSimulator, HiddenLearnerState, LearnerProfile
from .llm_contracts import compute_reward_from_interpreted
from .schemas import ACTION_BANK
from .stew_index import STEWFeatureIndex, build_stew_feature_index, load_feature_index, save_feature_index

__all__ = [
    "ACTION_BANK",
    "build_eeg_provider",
    "build_stew_feature_index",
    "DEFAULT_USER_TO_STEW_SUBJECT",
    "DecisionEngine",
    "EEG_FEATURE_NAMES",
    "EEGObservationContext",
    "EEGProvider",
    "EEGWindow",
    "HiddenLearnerSimulator",
    "HiddenLearnerState",
    "LearnerProfile",
    "load_feature_index",
    "MatchedRealEEGProvider",
    "save_feature_index",
    "SyntheticEEGProvider",
    "STEWFeatureIndex",
    "compute_reward_from_interpreted",
]

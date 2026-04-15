"""Validation environments for EEG-backed adaptive policy experiments."""

from .environment import NBackRestitchingEnvironment
from .reward import compute_nback_reward
from .windows import (
    NBACK_ACTION_BANK,
    NBACK_ACTION_IDS,
    NBACK_FEATURE_NAMES,
    ExperimentWindow,
    NBackObservation,
    NBackStateBuilder,
    build_toy_nback_windows,
)

__all__ = [
    "build_toy_nback_windows",
    "compute_nback_reward",
    "ExperimentWindow",
    "NBackObservation",
    "NBackRestitchingEnvironment",
    "NBackStateBuilder",
    "NBACK_ACTION_BANK",
    "NBACK_ACTION_IDS",
    "NBACK_FEATURE_NAMES",
]

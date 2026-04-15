from __future__ import annotations

import numpy as np

from emotiv_learn.eeg import EEG_SUMMARY_FEATURE_NAMES
from emotiv_learn.eeg_features import CHANNELS, EPOCH_SAMPLES
from emotiv_learn.stew_index import build_stew_feature_index


def test_build_stew_feature_index_from_single_subject(tmp_path) -> None:
    stew_dir = tmp_path / "stew"
    stew_dir.mkdir()

    samples = np.tile(np.linspace(0.0, 1.0, EPOCH_SAMPLES * 2, dtype=np.float32).reshape(-1, 1), (1, len(CHANNELS)))
    np.savetxt(stew_dir / "sub01_hi.txt", samples, delimiter=",")

    index = build_stew_feature_index(stew_dir=stew_dir, feature_names=EEG_SUMMARY_FEATURE_NAMES)

    assert index.feature_names == EEG_SUMMARY_FEATURE_NAMES
    assert "sub01" in index.windows_by_subject
    assert len(index.windows_by_subject["sub01"]) == 2

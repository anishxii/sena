from __future__ import annotations

from emotiv_learn.cog_bci_ingest import build_subject_cog_bci_nback_windows, list_available_subject_archives


def test_available_subject_archives_lists_local_subject() -> None:
    subjects = list_available_subject_archives("data/cog_bci")
    assert "sub-01" in subjects


def test_build_subject_cog_bci_nback_windows_from_local_archive() -> None:
    windows = build_subject_cog_bci_nback_windows(
        cog_bci_dir="data/cog_bci",
        subject_id="sub-01",
        window_sec=30,
        stride_sec=30,
    )

    assert windows
    sample = windows[0]
    assert sample.subject_id == "sub-01"
    assert sample.condition in {"ZeroBack", "OneBack", "TwoBack"}
    assert len(sample.eeg_features) == 8
    assert 0.0 <= sample.workload_estimate <= 1.0

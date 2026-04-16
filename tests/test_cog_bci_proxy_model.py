from __future__ import annotations

from emotiv_learn.cog_bci_ingest import build_subject_cog_bci_nback_windows
from emotiv_learn.cog_bci_proxy_model import (
    fit_cog_bci_proxy_regressor,
    load_cog_bci_proxy_regressor,
    save_cog_bci_proxy_regressor,
)


def test_fit_and_reload_cog_bci_proxy_regressor(tmp_path) -> None:
    windows = build_subject_cog_bci_nback_windows(
        cog_bci_dir="data/cog_bci",
        subject_id="sub-01",
        window_sec=30,
        stride_sec=30,
    )
    model = fit_cog_bci_proxy_regressor(windows)
    path = tmp_path / "cog_bci_proxy_model.json"
    save_cog_bci_proxy_regressor(model, path)
    loaded = load_cog_bci_proxy_regressor(path)

    prediction = loaded.predict(windows[0].eeg_features)
    assert set(prediction) == {
        "workload_estimate",
        "rolling_accuracy",
        "rolling_rt_percentile",
        "lapse_rate",
    }
    assert 0.0 <= prediction["workload_estimate"] <= 1.0

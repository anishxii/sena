from emotiv_learn.tutor_proxy import derive_tutor_facing_proxy_state


def test_tutor_proxy_reflects_overload_and_repair_pressure() -> None:
    proxy = derive_tutor_facing_proxy_state(
        interpreted={
            "followup_type": "clarify",
            "confusion_score": 0.85,
            "comprehension_score": 0.2,
            "engagement_score": 0.45,
            "progress_signal": 0.1,
            "pace_slow_score": 0.8,
            "checkpoint_score": 0.0,
        },
        student_response={"self_reported_confidence": 0.2},
        eeg_proxy_estimates={
            "workload_estimate": 0.9,
            "rolling_accuracy": 0.35,
            "rolling_rt_percentile": 0.88,
            "lapse_rate": 0.4,
        },
    )

    assert proxy.overload_risk > 0.75
    assert proxy.repair_need > 0.75
    assert proxy.challenge_readiness < 0.4


def test_tutor_proxy_reflects_branching_curiosity_and_checkpoint_readiness() -> None:
    proxy = derive_tutor_facing_proxy_state(
        interpreted={
            "followup_type": "branch",
            "confusion_score": 0.15,
            "comprehension_score": 0.82,
            "engagement_score": 0.9,
            "progress_signal": 0.75,
            "pace_slow_score": 0.05,
            "checkpoint_score": 1.0,
        },
        student_response={"self_reported_confidence": 0.8},
        eeg_proxy_estimates={
            "workload_estimate": 0.22,
            "rolling_accuracy": 0.91,
            "rolling_rt_percentile": 0.18,
            "lapse_rate": 0.05,
        },
    )

    assert proxy.curiosity_headroom > 0.7
    assert proxy.checkpoint_readiness > 0.7
    assert proxy.overload_risk < 0.4

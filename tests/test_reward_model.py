from emotiv_learn.reward_model import compute_observable_learning_reward, compute_reward_from_interpreted


def test_legacy_reward_penalizes_clarify_more_than_observable_reward() -> None:
    interpreted = {
        "followup_type": "clarify",
        "checkpoint_correct": None,
        "progress_signal": 0.6,
        "comprehension_score": 0.55,
        "engagement_score": 0.5,
        "confusion_score": 0.55,
        "pace_fast_score": 0.0,
        "pace_slow_score": 0.35,
    }

    assert compute_observable_learning_reward(interpreted) > compute_reward_from_interpreted(interpreted)


def test_observable_reward_prefers_correct_checkpoint() -> None:
    base = {
        "followup_type": "continue",
        "progress_signal": 0.4,
        "comprehension_score": 0.6,
        "engagement_score": 0.6,
        "confusion_score": 0.2,
        "pace_fast_score": 0.0,
        "pace_slow_score": 0.0,
    }

    correct = compute_observable_learning_reward({**base, "checkpoint_correct": True})
    incorrect = compute_observable_learning_reward({**base, "checkpoint_correct": False})

    assert correct > incorrect


def test_checkpoint_branch_gets_less_forward_progress_credit() -> None:
    base = {
        "followup_type": "branch",
        "checkpoint_correct": None,
        "progress_signal": 0.6,
        "comprehension_score": 0.6,
        "engagement_score": 0.8,
        "confusion_score": 0.2,
        "pace_fast_score": 0.0,
        "pace_slow_score": 0.0,
    }

    no_checkpoint = compute_observable_learning_reward({**base, "checkpoint_expected": False})
    skipped_checkpoint = compute_observable_learning_reward({**base, "checkpoint_expected": True})

    assert no_checkpoint > skipped_checkpoint

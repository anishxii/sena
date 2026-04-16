from emotiv_learn.decision_engine import DecisionEngine
from emotiv_learn.validation import (
    NBACK_ACTION_BANK,
    NBACK_FEATURE_NAMES,
    NBackRestitchingEnvironment,
    NBackStateBuilder,
    build_toy_nback_windows,
    compute_nback_reward,
)
from emotiv_learn.validation.runner import run_toy_nback_validation, summarize_validation


def test_toy_nback_windows_have_expected_shape() -> None:
    windows = build_toy_nback_windows(seed=3, subjects=2, windows_per_level=4)

    assert len(windows) == 24
    assert {window.difficulty_level for window in windows} == {0, 1, 2}
    assert all(0.0 <= window.workload_estimate <= 1.0 for window in windows)


def test_nback_state_builder_matches_feature_schema() -> None:
    env = NBackRestitchingEnvironment(build_toy_nback_windows(seed=2), max_turns=5, seed=2)
    observation = env.reset(subject_id="toy_sub01")

    state = NBackStateBuilder().build_state(observation)

    assert state.feature_names == NBACK_FEATURE_NAMES
    assert len(state.features) == len(NBACK_FEATURE_NAMES)


def test_reward_prefers_productive_challenge_over_overload() -> None:
    windows = build_toy_nback_windows(seed=4)
    productive = min(windows, key=lambda window: abs(window.workload_estimate - 0.55) + abs(window.rolling_accuracy - 0.80))
    overloaded = max(windows, key=lambda window: window.workload_estimate + window.lapse_rate)

    assert compute_nback_reward(productive) > compute_nback_reward(overloaded)


def test_environment_step_returns_action_conditioned_observation() -> None:
    env = NBackRestitchingEnvironment(build_toy_nback_windows(seed=5), max_turns=5, seed=5)
    observation = env.reset(subject_id="toy_sub02", difficulty_level=1)

    result = env.step("increase_difficulty")

    assert observation.window.difficulty_level == 1
    assert result.observation.window.difficulty_level == 2
    assert isinstance(result.reward, float)


def test_decision_engine_supports_validation_action_bank() -> None:
    env = NBackRestitchingEnvironment(build_toy_nback_windows(seed=6), max_turns=5, seed=6)
    observation = env.reset(subject_id="toy_sub01")
    state = NBackStateBuilder().build_state(observation)
    engine = DecisionEngine(
        feature_dim=len(state.features),
        action_bank=NBACK_ACTION_BANK,
        epsilon=0.0,
        seed=6,
    )

    action_scores = engine.score_actions(state, NBACK_ACTION_BANK)

    assert list(action_scores.scores) == [action.action_id for action in NBACK_ACTION_BANK]
    assert action_scores.selected_action == "maintain_difficulty"


def test_toy_validation_runner_returns_all_baselines() -> None:
    results = run_toy_nback_validation(turns=5, seed=8, subject_id="toy_sub02")
    summary = summarize_validation(results)

    assert set(results) == {"fixed_maintain", "classic_rule", "random", "learned_bandit"}
    assert set(summary) == set(results)
    assert all(len(payload["turn_logs"]) == 5 for payload in results.values())

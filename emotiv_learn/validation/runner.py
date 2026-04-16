from __future__ import annotations

from dataclasses import asdict

from emotiv_learn.decision_engine import DecisionEngine
from emotiv_learn.schemas import Outcome, RewardEvent, SemanticSignals, TaskResult

from .baselines import ClassicWorkloadRulePolicy, FixedMaintainPolicy, RandomNBackPolicy
from .cog_bci import build_subject_nback_windows
from .environment import NBackRestitchingEnvironment
from .windows import NBACK_ACTION_BANK, NBACK_FEATURE_NAMES, NBackStateBuilder, build_toy_nback_windows


def run_toy_nback_validation(turns: int = 30, seed: int = 7, subject_id: str = "toy_sub02") -> dict:
    windows = build_toy_nback_windows(seed=seed)
    return run_nback_validation(windows=windows, turns=turns, seed=seed, subject_id=subject_id)


def run_cog_bci_nback_validation(
    cog_bci_dir: str,
    turns: int = 30,
    seed: int = 7,
    subject_id: str = "sub-01",
) -> dict:
    windows = build_subject_nback_windows(cog_bci_dir=cog_bci_dir, subject_id=subject_id)
    return run_nback_validation(windows=windows, turns=turns, seed=seed, subject_id=subject_id)


def run_nback_validation(windows: list, turns: int, seed: int, subject_id: str) -> dict:
    return {
        "fixed_maintain": _run_baseline(FixedMaintainPolicy(), windows, turns, seed, subject_id),
        "classic_rule": _run_baseline(ClassicWorkloadRulePolicy(), windows, turns, seed, subject_id),
        "random": _run_baseline(RandomNBackPolicy(seed=seed), windows, turns, seed, subject_id),
        "learned_bandit": _run_learned_bandit(windows, turns, seed, subject_id),
    }


def summarize_validation(results: dict) -> dict:
    summary = {}
    for policy_name, payload in results.items():
        turns = payload["turn_logs"]
        summary[policy_name] = {
            "average_reward": sum(turn["reward"] for turn in turns) / len(turns),
            "average_workload": sum(turn["next_window"]["workload_estimate"] for turn in turns) / len(turns),
            "average_accuracy": sum(turn["next_window"]["rolling_accuracy"] for turn in turns) / len(turns),
            "overload_rate": sum(turn["next_window"]["workload_estimate"] > 0.78 for turn in turns) / len(turns),
            "action_counts": payload["action_counts"],
        }
    return summary


def _run_baseline(policy, windows, turns: int, seed: int, subject_id: str) -> dict:
    env = NBackRestitchingEnvironment(windows=windows, max_turns=turns, seed=seed)
    observation = env.reset(subject_id=subject_id)
    turn_logs = []
    action_counts = {action.action_id: 0 for action in NBACK_ACTION_BANK}
    for _ in range(turns):
        action_id = policy.select_action(observation.window)
        action_counts[action_id] += 1
        result = env.step(action_id)
        turn_logs.append(_turn_log(action_id, result.reward, result.observation, result.info))
        observation = result.observation
    return {"turn_logs": turn_logs, "action_counts": action_counts}


def _run_learned_bandit(windows, turns: int, seed: int, subject_id: str) -> dict:
    env = NBackRestitchingEnvironment(windows=windows, max_turns=turns, seed=seed)
    state_builder = NBackStateBuilder()
    engine = DecisionEngine(
        feature_dim=len(NBACK_FEATURE_NAMES),
        action_bank=NBACK_ACTION_BANK,
        epsilon=0.15,
        seed=seed,
        reward_clip_abs=1.0,
        update_clip_abs=0.15,
        l2_weight_decay=0.001,
    )
    observation = env.reset(subject_id=subject_id)
    turn_logs = []
    action_counts = {action.action_id: 0 for action in NBACK_ACTION_BANK}
    for _ in range(turns):
        state = state_builder.build_state(observation)
        action_scores = engine.score_actions(state, NBACK_ACTION_BANK)
        action = engine.select_action(action_scores)
        action_counts[action.action_id] += 1
        result = env.step(action.action_id)
        outcome = _make_outcome(result.observation, action.action_id)
        engine.update(
            RewardEvent(
                timestamp=result.observation.turn_index,
                user_id=state.user_id,
                state_features=state.features,
                action_id=action.action_id,
                reward=result.reward,
                outcome=outcome,
            )
        )
        turn_logs.append(
            {
                **_turn_log(action.action_id, result.reward, result.observation, result.info),
                "state": asdict(state),
                "action_scores": asdict(action_scores),
                "update_trace": asdict(engine.update_history[-1]),
            }
        )
        observation = result.observation
    return {"turn_logs": turn_logs, "action_counts": action_counts}


def _turn_log(action_id: str, reward: float, observation, info: dict) -> dict:
    return {
        "turn_index": observation.turn_index,
        "action_id": action_id,
        "reward": reward,
        "next_window": asdict(observation.window),
        "info": info,
    }


def _make_outcome(observation, action_id: str) -> Outcome:
    window = observation.window
    return Outcome(
        timestamp=observation.turn_index,
        user_id=window.subject_id,
        action_id=action_id,
        task_result=TaskResult(
            correct=round(window.rolling_accuracy),
            latency_s=window.rolling_rt_percentile,
            reread=None,
            completed=1,
            abandoned=0,
        ),
        semantic_signals=SemanticSignals(
            followup_text=None,
            followup_type="validation_window",
            confusion_score=None,
            comprehension_score=window.rolling_accuracy,
            engagement_score=1.0 - window.lapse_rate,
            pace_fast_score=None,
            pace_slow_score=window.rolling_rt_percentile,
        ),
        raw={"validation_window": asdict(window)},
    )

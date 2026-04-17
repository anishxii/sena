from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotiv_learn import ACTION_BANK, DecisionEngine
from emotiv_learn.eeg import EEGObservationContext, build_eeg_provider
from emotiv_learn.live_training import (
    LIVE_FEATURE_NAMES,
    STATE_PROFILE_CURRENT_EEG,
    STATE_PROFILES,
    LiveLLMStateBuilder,
    LiveStateInput,
)
from emotiv_learn.llm_contracts import (
    InterpreterPromptInput,
    StudentPromptInput,
    TutorPromptInput,
    build_interpreter_messages,
    build_student_messages,
    build_tutor_messages,
    normalize_interpreter_output,
    normalize_student_output,
)
from emotiv_learn.openai_client import OpenAIChatClient
from emotiv_learn.reward_model import compute_observable_learning_reward
from emotiv_learn.schemas import Outcome, RewardEvent, SemanticSignals, TaskResult
from emotiv_learn.student_model import HiddenKnowledgeStudent
from emotiv_learn.tutor_proxy import derive_tutor_facing_proxy_state
from scripts.knowledge_policy_comparison import (
    CONTENT_STEPS,
    USER_PROFILES,
    _initial_state_for_user,
)


POLICY_MODES = ["personalized", "generic", "fixed_no_change", "random"]

LEARNER_PROFILE = {
    "example_preference": 0.8,
    "challenge_preference": 0.4,
    "structure_preference": 0.75,
    "verbosity_tolerance": 0.45,
}

CHECKPOINT_RUBRICS = {
    "loss_function": (
        "A good answer says the loss function measures prediction error or mismatch between predictions and targets."
    ),
    "prediction_error": (
        "A good answer says prediction error is the difference between the model's output and the target."
    ),
    "activation_function": (
        "A good answer says activation functions add nonlinearity so neural networks can model more complex patterns."
    ),
    "gradient_descent_update": (
        "A good answer says the update subtracts learning_rate multiplied by the gradient "
        "from the current parameter."
    ),
    "backpropagation": (
        "A good answer says backpropagation moves gradients backward through layers so each parameter gets a learning signal."
    ),
    "momentum": (
        "A good answer says momentum carries forward part of the previous update direction to smooth or accelerate optimization."
    ),
    "vanishing_gradient": (
        "A good answer says vanishing gradients are very small gradient signals in deep networks that slow learning in earlier layers."
    ),
    "regularization": (
        "A good answer says regularization adds a penalty or constraint that encourages simpler models and can improve generalization."
    ),
    "convergence": (
        "A good answer connects convergence to repeated updates that move parameters "
        "toward lower loss or a stable minimum."
    ),
}

CHECKPOINT_ITEMS = {
    "loss_function": {
        "prompt": "What is the main role of the loss function during training?",
        "options": [
            "A. It measures how wrong the model's predictions are",
            "B. It sets the model's learning rate automatically",
            "C. It removes the need for gradients",
            "D. It stores the model parameters",
        ],
        "correct_choice": "A",
    },
    "prediction_error": {
        "prompt": "What is prediction error?",
        "options": [
            "A. The difference between the model output and the target",
            "B. The total number of model parameters",
            "C. The learning rate used by the optimizer",
            "D. The hidden-layer activation function",
        ],
        "correct_choice": "A",
    },
    "activation_function": {
        "prompt": "Why do neural networks use activation functions?",
        "options": [
            "A. To introduce nonlinearity into the network",
            "B. To remove the need for backpropagation",
            "C. To guarantee convergence in one step",
            "D. To keep all neuron outputs identical",
        ],
        "correct_choice": "A",
    },
    "gradient_descent_update": {
        "prompt": "Which expression best describes a gradient descent update?",
        "options": [
            "A. parameter_new = parameter_old - learning_rate * gradient",
            "B. parameter_new = parameter_old + learning_rate * gradient",
            "C. parameter_new = gradient - learning_rate",
            "D. parameter_new = parameter_old * gradient",
        ],
        "correct_choice": "A",
    },
    "backpropagation": {
        "prompt": "What does backpropagation mainly do?",
        "options": [
            "A. It propagates gradients backward so each layer can update its parameters",
            "B. It increases the learning rate after every update",
            "C. It replaces the loss function with a fixed rule",
            "D. It removes hidden layers from the network",
        ],
        "correct_choice": "A",
    },
    "momentum": {
        "prompt": "What is momentum used for in optimization?",
        "options": [
            "A. To smooth and accelerate updates using past gradient directions",
            "B. To remove gradients from the update rule entirely",
            "C. To force the loss to zero immediately",
            "D. To replace the learning rate with batch size",
        ],
        "correct_choice": "A",
    },
    "vanishing_gradient": {
        "prompt": "What is a vanishing gradient problem?",
        "options": [
            "A. Gradients become very small, so earlier layers learn slowly",
            "B. Gradients become infinite on every step",
            "C. The optimizer stops computing loss",
            "D. The model deletes hidden layers during training",
        ],
        "correct_choice": "A",
    },
    "regularization": {
        "prompt": "What is the main purpose of regularization?",
        "options": [
            "A. To encourage simpler models and improve generalization",
            "B. To make the learning rate always increase",
            "C. To guarantee zero training loss instantly",
            "D. To prevent gradients from being computed",
        ],
        "correct_choice": "A",
    },
    "convergence": {
        "prompt": "What happens as gradient descent converges?",
        "options": [
            "A. The loss usually decreases and parameter updates become smaller",
            "B. The learning rate always increases",
            "C. The gradient becomes random",
            "D. The model stops using gradients",
        ],
        "correct_choice": "A",
    },
}


def run_live_policy_comparison(
    *,
    turns: int,
    seed: int,
    model: str | None,
    output_path: Path,
    events_output_path: Path | None = None,
    policy_modes: list[str] | None = None,
    user_ids: list[str] | None = None,
    eeg_mode: str = "synthetic",
    state_profile: str = STATE_PROFILE_CURRENT_EEG,
) -> dict:
    client = OpenAIChatClient(model=model)
    selected_modes = policy_modes or POLICY_MODES
    selected_users = user_ids or list(USER_PROFILES)
    event_writer = JsonlEventWriter(events_output_path)
    results: dict[str, dict] = {}
    event_writer.write(
        "experiment_started",
        {
            "turns": turns,
            "seed": seed,
            "model": model,
            "policy_modes": selected_modes,
            "user_ids": selected_users,
        },
    )

    for policy_mode in selected_modes:
        print(f"=== Running {policy_mode} ===")
        event_writer.write("policy_mode_started", {"policy_mode": policy_mode})
        results[policy_mode] = run_policy_mode_live(
            policy_mode=policy_mode,
            turns=turns,
            seed=seed,
            user_ids=selected_users,
            client=client,
            event_writer=event_writer,
            eeg_mode=eeg_mode,
            state_profile=state_profile,
        )
        event_writer.write(
            "policy_mode_completed",
            {
                "policy_mode": policy_mode,
                "summary": summarize_policy(results[policy_mode]["users"]),
            },
        )

    output = {
        "turns": turns,
        "seed": seed,
        "model": model,
        "state_profile": state_profile,
        "results": results,
        "summary": {policy_mode: summarize_policy(rows["users"]) for policy_mode, rows in results.items()},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    event_writer.write(
        "experiment_completed",
        {
            "output_path": str(output_path),
            "summary": output["summary"],
        },
    )
    print(f"Wrote live policy comparison to {output_path}")
    print_results(output["summary"], results)
    return output


def run_policy_mode_live(
    *,
    policy_mode: str,
    turns: int,
    seed: int,
    user_ids: list[str],
    client: OpenAIChatClient,
    event_writer: JsonlEventWriter,
    eeg_mode: str,
    state_profile: str,
) -> dict:
    rng = random.Random(seed)
    state_builder = LiveLLMStateBuilder(state_profile=state_profile)
    eeg_provider = build_eeg_provider(
        eeg_mode=eeg_mode,
        seed=seed + 1000,
    )
    students = {
        user_id: HiddenKnowledgeStudent(_initial_state_for_user(user_id), seed=seed + index * 101)
        for index, user_id in enumerate(user_ids)
    }
    engine = None
    if policy_mode in {"personalized", "generic"}:
        engine = DecisionEngine(
            feature_dim=len(LIVE_FEATURE_NAMES),
            epsilon=0.22,
            use_personalization=policy_mode == "personalized",
            seed=seed,
            reward_clip_abs=1.5,
            update_clip_abs=0.2,
            l2_weight_decay=0.001,
        )

    trackers = {user_id: _new_tracker() for user_id in user_ids}
    turn_logs: list[dict] = []

    for turn_index in range(1, turns + 1):
        concept_id, checkpoint_expected = CONTENT_STEPS[(turn_index - 1) % len(CONTENT_STEPS)]
        for user_id, student in students.items():
            tracker = trackers[user_id]
            state = state_builder.build_state(
                LiveStateInput(
                    timestamp=turn_index,
                    user_id=user_id,
                    topic_id="gradient_descent",
                    task_type="learn",
                    difficulty="medium",
                    turn_index=turn_index,
                    max_turns=turns,
                    interpreted=tracker["previous_interpreted"],
                    student_response=tracker["previous_student_response"],
                    previous_reward=tracker["previous_reward"],
                    eeg_features=tracker.get("previous_eeg_features"),
                    eeg_proxy_estimates=tracker.get("previous_eeg_proxy"),
                )
            )
            action_id = _select_action(policy_mode, engine, state, rng)
            event_writer.write(
                "turn_started",
                {
                    "policy_mode": policy_mode,
                    "turn_index": turn_index,
                    "user_id": user_id,
                    "concept_id": concept_id,
                    "checkpoint_expected": checkpoint_expected,
                    "action_id": action_id,
                    "state": asdict(state),
                },
            )
            tutor_message = _generate_tutor_message(
                client=client,
                user_id=user_id,
                concept_id=concept_id,
                action_id=action_id,
                checkpoint_expected=checkpoint_expected,
                tracker=tracker,
            )
            event_writer.write(
                "tutor_message",
                {
                    "policy_mode": policy_mode,
                    "turn_index": turn_index,
                    "user_id": user_id,
                    "action_id": action_id,
                    "tutor_message": tutor_message,
                },
            )
            transition = student.step(
                concept_id=concept_id,
                action_id=action_id,
                tutor_message=tutor_message,
                checkpoint_expected=checkpoint_expected,
            )
            eeg_window = eeg_provider.observe(
                EEGObservationContext(
                    timestamp=turn_index,
                    user_id=user_id,
                    concept_id=concept_id,
                    action_id=action_id,
                    tutor_message=tutor_message,
                    time_on_chunk=None,
                    hidden_state=transition.to_dict()["hidden_state_after"],
                    observable_signals=transition.observable_signals,
                )
            )
            eeg_proxy = dict(eeg_window.metadata.get("proxy_state", {}))
            event_writer.write(
                "student_transition",
                {
                    "policy_mode": policy_mode,
                    "turn_index": turn_index,
                    "user_id": user_id,
                    "action_id": action_id,
                    "student_transition": transition.to_dict(),
                    "eeg_window": _serialize_eeg_window(eeg_window),
                    "eeg_proxy_estimate": eeg_proxy,
                },
            )
            student_response = _generate_student_response(
                client=client,
                concept_id=concept_id,
                transition=transition,
                tutor_message=tutor_message,
                checkpoint_expected=checkpoint_expected,
            )
            interpreted = _generate_interpreted_outcome(
                client=client,
                concept_id=concept_id,
                action_id=action_id,
                checkpoint_expected=checkpoint_expected,
                tracker=tracker,
                tutor_message=tutor_message,
                student_response=student_response,
            )
            interpreted["checkpoint_expected"] = checkpoint_expected
            tutor_proxy = derive_tutor_facing_proxy_state(
                interpreted=interpreted,
                student_response=student_response,
                eeg_proxy_estimates=eeg_proxy,
            ).as_feature_dict()
            reward = _compute_live_reward(action_id=action_id, tracker=tracker, interpreted=interpreted)

            if engine is not None:
                outcome = _make_outcome(turn_index, user_id, action_id, interpreted, student_response, tutor_message)
                engine.update(
                    RewardEvent(
                        timestamp=turn_index,
                        user_id=user_id,
                        state_features=state.features,
                        action_id=action_id,
                        reward=reward,
                        outcome=outcome,
                    )
                )
                update_trace = asdict(engine.update_history[-1])
            else:
                update_trace = None

            _update_tracker(
                tracker=tracker,
                action_id=action_id,
                transition=transition,
                interpreted=interpreted,
                student_response=student_response,
                reward=reward,
                eeg_window=eeg_window,
                eeg_proxy=eeg_proxy,
            )
            turn_logs.append(
                {
                    "policy_mode": policy_mode,
                    "turn_index": turn_index,
                    "user_id": user_id,
                    "state": asdict(state),
                    "action_id": action_id,
                    "tutor_message": tutor_message,
                    "student_response": student_response,
                    "interpreted": interpreted,
                    "reward": reward,
                    "eeg_window": _serialize_eeg_window(eeg_window),
                    "eeg_proxy_estimate": eeg_proxy,
                    "tutor_proxy_estimate": tutor_proxy,
                    "student_transition": transition.to_dict(),
                    "update_trace": update_trace,
                }
            )
            event_writer.write(
                "turn_completed",
                {
                    "policy_mode": policy_mode,
                    "turn_index": turn_index,
                    "user_id": user_id,
                    "concept_id": concept_id,
                    "checkpoint_expected": checkpoint_expected,
                    "action_id": action_id,
                    "tutor_message": tutor_message,
                    "student_response": student_response,
                    "interpreted": interpreted,
                    "reward": reward,
                    "eeg_window": _serialize_eeg_window(eeg_window),
                    "eeg_proxy_estimate": eeg_proxy,
                    "tutor_proxy_estimate": tutor_proxy,
                    "oracle_mastery_gain": transition.oracle_mastery_gain,
                    "response_type": transition.sampled_response_type,
                    "checkpoint_correct": transition.checkpoint_correct,
                    "student_transition": transition.to_dict(),
                    "update_trace": update_trace,
                },
            )
            content_mix = _content_mix_from_transition(transition)
            print(
                f"{policy_mode} turn={turn_index} user={user_id} action={action_id} "
                f"mix={_dominant_content_mix(content_mix)} "
                f"(d={content_mix['text_description']:.2f}, "
                f"e={content_mix['text_examples']:.2f}, "
                f"v={content_mix['visual']:.2f}) "
                f"followup={transition.sampled_response_type} reward={reward:.3f} "
                f"oracle_gain={transition.oracle_mastery_gain:.3f}"
            )

    return {
        "users": [_summarize_user(user_id, trackers[user_id], turns) for user_id in user_ids],
        "turn_logs": turn_logs,
    }


def _new_tracker() -> dict:
    return {
        "previous_interpreted": None,
        "previous_student_response": None,
        "previous_reward": 0.0,
        "previous_eeg_features": None,
        "previous_eeg_proxy": None,
        "previous_action_id": None,
        "conversation_summary": "The learner is beginning a short lesson.",
        "total_reward": 0.0,
        "total_oracle_gain": 0.0,
        "checkpoint_correct": 0,
        "checkpoint_count": 0,
        "followups": Counter(),
        "action_counts": Counter(),
    }


class JsonlEventWriter:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.event_index = 0
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def write(self, event_type: str, payload: dict) -> None:
        if self.path is None:
            return
        self.event_index += 1
        event = {
            "event_id": f"evt_{self.event_index:06d}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")


def _serialize_eeg_window(eeg_window) -> dict:
    payload = asdict(eeg_window)
    payload.pop("channels", None)
    return payload


def _state_summary(interpreted: dict | None) -> str:
    if not interpreted:
        return "first turn; no prior interpreted outcome"
    return json.dumps(interpreted, indent=2)


def _deterministic_checkpoint_choice(*, concept_id: str, transition) -> str | None:
    item = CHECKPOINT_ITEMS.get(concept_id)
    if item is None or transition.checkpoint_correct is None:
        return None
    if transition.checkpoint_correct:
        return item["correct_choice"]
    for option in ["A", "B", "C", "D"]:
        if option != item["correct_choice"]:
            return option
    return None


def _score_checkpoint_choice(
    *,
    concept_id: str,
    checkpoint_expected: bool,
    checkpoint_choice: str | None,
) -> tuple[bool | None, float | None]:
    if not checkpoint_expected:
        return None, None
    item = CHECKPOINT_ITEMS.get(concept_id)
    if item is None or checkpoint_choice is None:
        return None, None
    normalized = str(checkpoint_choice).strip().upper()
    correct = normalized == item["correct_choice"]
    return correct, 1.0 if correct else 0.0


def _enforce_checkpoint_schedule(interpreted: dict, checkpoint_expected: bool) -> dict:
    if checkpoint_expected:
        return interpreted
    corrected = dict(interpreted)
    corrected["checkpoint_correct"] = None
    corrected["checkpoint_score"] = None
    if corrected["followup_type"] == "checkpoint_answer":
        corrected["followup_type"] = "mixed"
    return corrected


def _generate_tutor_message(
    *,
    client: OpenAIChatClient,
    user_id: str,
    concept_id: str,
    action_id: str,
    checkpoint_expected: bool,
    tracker: dict,
) -> str:
    checkpoint_item = CHECKPOINT_ITEMS.get(concept_id) if checkpoint_expected else None
    messages = build_tutor_messages(
        TutorPromptInput(
            topic="gradient descent",
            concept_id=concept_id,
            conversation_summary=tracker["conversation_summary"],
            load_level=_load_level(tracker["previous_interpreted"]),
            behavior_summary=_behavior_summary(tracker["previous_interpreted"]),
            last_followup_type=(tracker["previous_interpreted"] or {}).get("followup_type", "unknown"),
            action_id=action_id,
            length_target="short",
            difficulty_target="medium",
            include_checkpoint=checkpoint_expected,
            learner_format_hint=_learner_format_hint(user_id),
            checkpoint_prompt=None if checkpoint_item is None else checkpoint_item["prompt"],
            checkpoint_options=None if checkpoint_item is None else checkpoint_item["options"],
        )
    )
    return client.complete_text(messages, max_tokens=700, temperature=0.35)


def _generate_student_response(
    *,
    client: OpenAIChatClient,
    concept_id: str,
    transition,
    tutor_message: str,
    checkpoint_expected: bool,
) -> dict:
    checkpoint_item = CHECKPOINT_ITEMS.get(concept_id) if checkpoint_expected else None
    checkpoint_choice = _deterministic_checkpoint_choice(concept_id=concept_id, transition=transition)
    messages = build_student_messages(
        StudentPromptInput(
            learner_profile=LEARNER_PROFILE,
            hidden_state=transition.to_dict()["hidden_state_after"],
            observable_signals=transition.observable_signals,
            tutor_message=tutor_message,
            checkpoint_expected=checkpoint_expected,
            sampled_response_type=transition.sampled_response_type,
            checkpoint_answer=transition.checkpoint_answer,
            checkpoint_prompt=None if checkpoint_item is None else checkpoint_item["prompt"],
            checkpoint_options=None if checkpoint_item is None else checkpoint_item["options"],
            checkpoint_choice=checkpoint_choice,
        )
    )
    student_response = normalize_student_output(client.complete_json(messages, max_tokens=500, temperature=0.45))
    student_response["response_type"] = transition.sampled_response_type
    if transition.checkpoint_answer is not None:
        student_response["checkpoint_answer"] = transition.checkpoint_answer
    student_response["checkpoint_choice"] = checkpoint_choice
    return student_response


def _learner_format_hint(user_id: str) -> str:
    profile = USER_PROFILES[user_id]
    learning_style = profile.get("learning_style", {})
    text_description = float(learning_style.get("text_description", 0.5))
    text_examples = float(learning_style.get("text_examples", 0.3))
    visual = float(learning_style.get("visual", 0.2))

    if visual >= max(text_description, text_examples):
        return (
            "Use scan-friendly structure. This is a hard formatting requirement: no dense paragraphs. "
            "Write exactly 3 to 5 short labeled sections, include at least one bulleted or numbered list, "
            "and include at least one arrow trace like a -> b -> c."
        )
    if text_examples >= text_description:
        return (
            "Teach example-first: begin with a concrete worked example or numeric trace, "
            "then connect it back to the concept in one short explanation."
        )
    return (
        "Teach description-first: begin with a concise conceptual explanation, then add at most "
        "one short example if needed."
    )


def _generate_interpreted_outcome(
    *,
    client: OpenAIChatClient,
    concept_id: str,
    action_id: str,
    checkpoint_expected: bool,
    tracker: dict,
    tutor_message: str,
    student_response: dict,
) -> dict:
    messages = build_interpreter_messages(
        InterpreterPromptInput(
            tutor_message=tutor_message,
            student_response=student_response,
            checkpoint_rubric=CHECKPOINT_RUBRICS.get(concept_id) if checkpoint_expected else None,
            topic="gradient descent",
            concept_id=concept_id,
            action_id=action_id,
            state_summary=_state_summary(tracker["previous_interpreted"]),
        )
    )
    interpreted = normalize_interpreter_output(client.complete_json(messages, max_tokens=500, temperature=0.1))
    deterministic_correct, deterministic_score = _score_checkpoint_choice(
        concept_id=concept_id,
        checkpoint_expected=checkpoint_expected,
        checkpoint_choice=student_response.get("checkpoint_choice"),
    )
    interpreted["checkpoint_correct"] = deterministic_correct
    interpreted["checkpoint_score"] = deterministic_score
    return _enforce_checkpoint_schedule(interpreted, checkpoint_expected=checkpoint_expected)


def _compute_live_reward(*, action_id: str, tracker: dict, interpreted: dict) -> float:
    reward = compute_observable_learning_reward(interpreted)
    if action_id == "no_change":
        if interpreted["progress_signal"] < 0.45 and interpreted["comprehension_score"] < 0.60:
            reward -= 0.12
        if tracker.get("previous_action_id") == "no_change" and interpreted["progress_signal"] < 0.55:
            reward -= 0.08
        if interpreted["followup_type"] == "clarify":
            reward -= 0.05
    return max(-1.5, min(1.5, reward))


def _update_tracker(
    *,
    tracker: dict,
    action_id: str,
    transition,
    interpreted: dict,
    student_response: dict,
    reward: float,
    eeg_window,
    eeg_proxy: dict | None,
) -> None:
    tracker["total_reward"] += reward
    tracker["total_oracle_gain"] += transition.oracle_mastery_gain
    tracker["action_counts"][action_id] += 1
    tracker["followups"][transition.sampled_response_type] += 1
    if transition.checkpoint_correct is not None:
        tracker["checkpoint_count"] += 1
        tracker["checkpoint_correct"] += int(transition.checkpoint_correct)
    tracker["previous_interpreted"] = interpreted
    tracker["previous_student_response"] = student_response
    tracker["previous_reward"] = reward
    tracker["previous_eeg_features"] = eeg_window.features
    tracker["previous_eeg_proxy"] = eeg_proxy
    tracker["previous_action_id"] = action_id
    tracker["conversation_summary"] = (
        f"Previous action {action_id}; student responded {interpreted['followup_type']} "
        f"with reward {reward:.2f} and oracle mastery gain {transition.oracle_mastery_gain:.3f}."
    )


def _select_action(policy_mode: str, engine: DecisionEngine | None, state, rng: random.Random) -> str:
    if policy_mode == "fixed_no_change":
        return "no_change"
    if policy_mode == "random":
        return rng.choice([action.action_id for action in ACTION_BANK])
    if engine is None:
        raise ValueError(f"policy mode requires engine: {policy_mode}")
    action_scores = engine.score_actions(state, ACTION_BANK)
    return engine.select_action(action_scores).action_id


def _transition_to_interpreted(transition) -> dict:
    observables = transition.observable_signals
    return {
        "followup_type": transition.sampled_response_type,
        "checkpoint_correct": transition.checkpoint_correct,
        "checkpoint_score": observables["checkpoint_score"],
        "confusion_score": observables["confusion_score"],
        "comprehension_score": observables["comprehension_score"],
        "engagement_score": observables["engagement_score"],
        "progress_signal": observables["progress_signal"],
        "pace_fast_score": observables["pace_fast_score"],
        "pace_slow_score": observables["pace_slow_score"],
        "evidence": {
            "confusion_phrases": [],
            "understanding_phrases": [],
            "curiosity_phrases": [],
        },
    }


def _make_outcome(
    turn_index: int,
    user_id: str,
    action_id: str,
    interpreted: dict,
    student_response: dict,
    tutor_message: str,
 ) -> Outcome:
    checkpoint_correct = interpreted["checkpoint_correct"]
    return Outcome(
        timestamp=turn_index,
        user_id=user_id,
        action_id=action_id,
        task_result=TaskResult(
            correct=None if checkpoint_correct is None else int(checkpoint_correct),
            latency_s=None,
            reread=None,
            completed=1,
            abandoned=0,
        ),
        semantic_signals=SemanticSignals(
            followup_text=student_response.get("student_message"),
            followup_type=interpreted["followup_type"],
            confusion_score=interpreted["confusion_score"],
            comprehension_score=interpreted["comprehension_score"],
            engagement_score=interpreted["engagement_score"],
            pace_fast_score=interpreted["pace_fast_score"],
            pace_slow_score=interpreted["pace_slow_score"],
        ),
        raw={
            "student_response": student_response,
            "interpreted": interpreted,
            "tutor_message": tutor_message,
        },
    )


def _summarize_user(user_id: str, tracker: dict, turns: int) -> dict:
    return {
        "user_id": user_id,
        "average_reward": tracker["total_reward"] / turns,
        "total_oracle_mastery_gain": tracker["total_oracle_gain"],
        "average_oracle_mastery_gain": tracker["total_oracle_gain"] / turns,
        "checkpoint_accuracy": (
            tracker["checkpoint_correct"] / tracker["checkpoint_count"] if tracker["checkpoint_count"] else 0.0
        ),
        "followups": dict(tracker["followups"]),
        "action_counts": {action.action_id: tracker["action_counts"][action.action_id] for action in ACTION_BANK},
    }


def _content_mix_from_transition(transition) -> dict[str, float]:
    evaluation = transition.evaluation
    return {
        "text_description": float(evaluation.get("content_text_description", 0.0)),
        "text_examples": float(evaluation.get("content_text_examples", 0.0)),
        "visual": float(evaluation.get("content_visual", 0.0)),
    }


def _dominant_content_mix(content_mix: dict[str, float]) -> str:
    return max(content_mix, key=content_mix.get)


def summarize_policy(rows: list[dict]) -> dict:
    return {
        "average_reward": sum(row["average_reward"] for row in rows) / len(rows),
        "total_oracle_mastery_gain": sum(row["total_oracle_mastery_gain"] for row in rows) / len(rows),
        "average_oracle_mastery_gain": sum(row["average_oracle_mastery_gain"] for row in rows) / len(rows),
        "checkpoint_accuracy": sum(row["checkpoint_accuracy"] for row in rows) / len(rows),
    }


def print_results(summary: dict[str, dict], results: dict[str, dict]) -> None:
    for policy_mode, policy_summary in summary.items():
        print(f"=== {policy_mode} ===")
        print(
            "avg_reward={average_reward:.3f} total_oracle_gain={total_oracle_mastery_gain:.3f} "
            "avg_oracle_gain={average_oracle_mastery_gain:.3f} checkpoint_acc={checkpoint_accuracy:.3f}".format(
                **policy_summary
            )
        )
        for row in results[policy_mode]["users"]:
            top_action = max(row["action_counts"], key=row["action_counts"].get)
            print(
                f"  {row['user_id']}: reward={row['average_reward']:.3f} "
                f"oracle_gain={row['total_oracle_mastery_gain']:.3f} "
                f"checkpoint_acc={row['checkpoint_accuracy']:.3f} "
                f"top_action={top_action}"
            )
        print()

    if "fixed_no_change" in summary:
        baseline = summary["fixed_no_change"]
        print("=== Lift vs fixed_no_change ===")
        for policy_mode in summary:
            if policy_mode == "fixed_no_change":
                continue
            print(
                f"{policy_mode}: reward_delta={summary[policy_mode]['average_reward'] - baseline['average_reward']:+.3f} "
                f"oracle_gain_delta={summary[policy_mode]['total_oracle_mastery_gain'] - baseline['total_oracle_mastery_gain']:+.3f}"
            )

    if "personalized" in summary and "generic" in summary:
        print("=== Lift: personalized - generic ===")
        print(
            f"reward_delta={summary['personalized']['average_reward'] - summary['generic']['average_reward']:+.3f} "
            f"oracle_gain_delta={summary['personalized']['total_oracle_mastery_gain'] - summary['generic']['total_oracle_mastery_gain']:+.3f}"
        )


def _load_level(interpreted: dict | None) -> str:
    if not interpreted:
        return "unknown"
    confusion = interpreted.get("confusion_score", 0.5)
    if confusion >= 0.65:
        return "high"
    if confusion >= 0.35:
        return "medium"
    return "low"


def _behavior_summary(interpreted: dict | None) -> str:
    if not interpreted:
        return "no prior behavioral signal"
    return (
        f"confusion={interpreted['confusion_score']:.2f}, "
        f"comprehension={interpreted['comprehension_score']:.2f}, "
        f"engagement={interpreted['engagement_score']:.2f}"
    )


def _parse_csv(value: str | None, allowed: set[str]) -> list[str] | None:
    if not value:
        return None
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in parsed if item not in allowed]
    if unknown:
        raise ValueError(f"unknown values {unknown}; allowed values are {sorted(allowed)}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live LLM policy comparison on hidden-knowledge students.")
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model", default=None)
    parser.add_argument("--modes", default="personalized,generic,fixed_no_change,random")
    parser.add_argument("--users", default=",".join(USER_PROFILES))
    parser.add_argument("--output", default="artifacts/live_policy_comparison.json")
    parser.add_argument("--events-output", default=None)
    parser.add_argument("--eeg-mode", default="synthetic", choices=["synthetic"])
    parser.add_argument("--state-profile", default=STATE_PROFILE_CURRENT_EEG, choices=STATE_PROFILES)
    args = parser.parse_args()

    run_live_policy_comparison(
        turns=args.turns,
        seed=args.seed,
        model=args.model,
        output_path=Path(args.output),
        events_output_path=Path(args.events_output) if args.events_output else None,
        policy_modes=_parse_csv(args.modes, set(POLICY_MODES)),
        user_ids=_parse_csv(args.users, set(USER_PROFILES)),
        eeg_mode=args.eeg_mode,
        state_profile=args.state_profile,
    )


if __name__ == "__main__":
    main()

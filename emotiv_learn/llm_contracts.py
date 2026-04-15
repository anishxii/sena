from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


ACTION_INSTRUCTIONS = {
    "no_change": "Continue in the same style and difficulty. Do not add unnecessary intervention.",
    "simplify": "Explain the same concept using simpler language, shorter sentences, and reduced abstraction.",
    "deepen": "Add more conceptual or technical depth. Assume the learner is ready for a challenge.",
    "summarize": "Condense the key idea into a shorter explanation that preserves the core meaning.",
    "highlight_key_points": "Emphasize the most important takeaways using clear structure and salient bullets.",
    "worked_example": "Teach through a concrete worked example with clear steps.",
    "analogy": "Use a helpful analogy or metaphor, then connect it back to the original concept.",
    "step_by_step": "Break the explanation into small sequential steps with minimal jumps.",
}

TUTOR_SYSTEM_PROMPT = """You are an adaptive AI tutor inside Emotiv Learn.

Your job is to teach the learner the current concept using the provided teaching strategy.

You must follow the teaching strategy exactly. Do not mention the strategy name. Do not mention EEG, reward, policy, personalization, or internal state.

Keep the response focused on the current concept. Do not jump ahead unless the strategy explicitly asks for deeper exploration.

The learner should feel supported, not evaluated, unless a checkpoint is explicitly requested.

Return only the tutor-facing content."""

STUDENT_SYSTEM_PROMPT = """You are simulating a student using an adaptive learning app.

You must respond as the student, not as a tutor or evaluator.

Your response must be consistent with the provided learner profile, current learning state, and the tutor message.

Do not mention hidden state variables, EEG, policy, reward, or simulation.

You may respond in one of three modes:
- continue: the student is ready to move on
- clarify: the student is confused or needs re-explanation
- branch: the student is curious and asks a related follow-up question

If a checkpoint question is asked, answer it as the student would, given the current mastery and confusion levels. The answer may be correct, partially correct, or incorrect.

Return valid JSON only."""

INTERPRETER_SYSTEM_PROMPT = """You are an outcome interpreter for an adaptive learning experiment.

Your job is to convert the tutor-student interaction into structured outcome signals.

Do not reward the policy directly. Do not assign a final scalar reward. Only extract signals.

Be conservative. If evidence is unclear, use moderate scores rather than extreme scores.

Use the checkpoint rubric if provided. If no checkpoint was asked, checkpoint_correct must be null.

Return valid JSON only."""


@dataclass(frozen=True)
class TutorPromptInput:
    topic: str
    concept_id: str
    conversation_summary: str
    load_level: str
    behavior_summary: str
    last_followup_type: str
    action_id: str
    length_target: str
    difficulty_target: str
    include_checkpoint: bool


@dataclass(frozen=True)
class StudentPromptInput:
    learner_profile: dict[str, Any]
    hidden_state: dict[str, float]
    observable_signals: dict[str, Any]
    tutor_message: str


@dataclass(frozen=True)
class InterpreterPromptInput:
    tutor_message: str
    student_response: dict[str, Any]
    checkpoint_rubric: str | None
    topic: str
    concept_id: str
    action_id: str
    state_summary: str


def build_tutor_messages(prompt_input: TutorPromptInput) -> list[dict[str, str]]:
    action_instruction = ACTION_INSTRUCTIONS[prompt_input.action_id]
    user_prompt = f"""Topic:
{prompt_input.topic}

Current concept:
{prompt_input.concept_id}

Learner-visible prior context:
{prompt_input.conversation_summary}

Current learner state summary:
- estimated cognitive load: {prompt_input.load_level}
- recent behavior: {prompt_input.behavior_summary}
- recent learner response type: {prompt_input.last_followup_type}

Teaching strategy:
{prompt_input.action_id}

Teaching strategy instructions:
{action_instruction}

Content requirements:
- length target: {prompt_input.length_target}
- difficulty target: {prompt_input.difficulty_target}
- include checkpoint: {prompt_input.include_checkpoint}
- if checkpoint is true, end with one short comprehension question

Generate the next tutor message."""
    return [
        {"role": "system", "content": TUTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_student_messages(prompt_input: StudentPromptInput) -> list[dict[str, str]]:
    user_prompt = f"""Learner profile:
{json.dumps(prompt_input.learner_profile, indent=2)}

Current hidden learner state:
- mastery: {prompt_input.hidden_state["mastery"]}
- confusion: {prompt_input.hidden_state["confusion"]}
- curiosity: {prompt_input.hidden_state["curiosity"]}
- fatigue: {prompt_input.hidden_state["fatigue"]}
- engagement: {prompt_input.hidden_state["engagement"]}

Observable learner signals:
{json.dumps(prompt_input.observable_signals, indent=2)}

Tutor message:
{prompt_input.tutor_message}

If there is a checkpoint question, answer it naturally as this student.

Choose the most realistic learner response mode:
- continue
- clarify
- branch

Return JSON in this exact schema:
{{
  "response_type": "continue | clarify | branch",
  "student_message": "string or null",
  "checkpoint_answer": "string or null",
  "self_reported_confidence": 0.0,
  "rationale_for_simulation": "brief private explanation of why this response matches the learner state"
}}"""
    return [
        {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_interpreter_messages(prompt_input: InterpreterPromptInput) -> list[dict[str, str]]:
    user_prompt = f"""Tutor message:
{prompt_input.tutor_message}

Student response:
{json.dumps(prompt_input.student_response, indent=2)}

Expected checkpoint answer or rubric:
{prompt_input.checkpoint_rubric}

Context:
- topic: {prompt_input.topic}
- concept_id: {prompt_input.concept_id}
- selected teaching action: {prompt_input.action_id}
- prior learner state summary: {prompt_input.state_summary}

Extract structured outcome signals.

Return JSON in this exact schema:
{{
  "followup_type": "continue | clarify | branch | checkpoint_answer | mixed | unknown",
  "checkpoint_correct": true,
  "checkpoint_score": 0.0,
  "confusion_score": 0.0,
  "comprehension_score": 0.0,
  "engagement_score": 0.0,
  "progress_signal": 0.0,
  "pace_fast_score": 0.0,
  "pace_slow_score": 0.0,
  "evidence": {{
    "confusion_phrases": [],
    "understanding_phrases": [],
    "curiosity_phrases": []
  }}
}}"""
    return [
        {"role": "system", "content": INTERPRETER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def parse_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    parsed = json.loads(text.strip())
    if not isinstance(parsed, dict):
        raise ValueError("LLM output must parse to a JSON object")
    return parsed


def normalize_student_output(value: dict[str, Any]) -> dict[str, Any]:
    response_type = value.get("response_type")
    if response_type not in {"continue", "clarify", "branch"}:
        response_type = "clarify"
    confidence = _clip01(float(value.get("self_reported_confidence", 0.5)))
    return {
        "response_type": response_type,
        "student_message": value.get("student_message"),
        "checkpoint_answer": value.get("checkpoint_answer"),
        "self_reported_confidence": confidence,
        "rationale_for_simulation": value.get("rationale_for_simulation", ""),
    }


def normalize_interpreter_output(value: dict[str, Any]) -> dict[str, Any]:
    followup_type = value.get("followup_type")
    if followup_type not in {"continue", "clarify", "branch", "checkpoint_answer", "mixed", "unknown"}:
        followup_type = "unknown"

    checkpoint_correct = value.get("checkpoint_correct")
    if checkpoint_correct not in (True, False, None):
        checkpoint_correct = None

    evidence = value.get("evidence") if isinstance(value.get("evidence"), dict) else {}
    return {
        "followup_type": followup_type,
        "checkpoint_correct": checkpoint_correct,
        "checkpoint_score": _optional_score(value.get("checkpoint_score")),
        "confusion_score": _clip01(float(value.get("confusion_score", 0.5))),
        "comprehension_score": _clip01(float(value.get("comprehension_score", 0.5))),
        "engagement_score": _clip01(float(value.get("engagement_score", 0.5))),
        "progress_signal": _clip01(float(value.get("progress_signal", 0.5))),
        "pace_fast_score": _clip01(float(value.get("pace_fast_score", 0.0))),
        "pace_slow_score": _clip01(float(value.get("pace_slow_score", 0.0))),
        "evidence": {
            "confusion_phrases": list(evidence.get("confusion_phrases", [])),
            "understanding_phrases": list(evidence.get("understanding_phrases", [])),
            "curiosity_phrases": list(evidence.get("curiosity_phrases", [])),
        },
    }


def compute_reward_from_interpreted(interpreted: dict[str, Any]) -> float:
    reward = 0.0
    if interpreted["checkpoint_correct"] is True:
        reward += 1.0
    elif interpreted["checkpoint_correct"] is False:
        reward -= 0.8

    reward += 0.6 * interpreted["progress_signal"]
    reward += 0.4 * interpreted["comprehension_score"]
    reward += 0.2 * interpreted["engagement_score"]
    reward += 0.3 * int(interpreted["followup_type"] == "branch")
    reward += 0.1 * int(interpreted["followup_type"] == "continue")
    reward -= 0.6 * interpreted["confusion_score"]
    reward -= 0.3 * interpreted["pace_slow_score"]
    reward -= 0.2 * interpreted["pace_fast_score"]
    return max(-1.5, min(1.5, reward))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _optional_score(value: Any) -> float | None:
    if value is None:
        return None
    return _clip01(float(value))

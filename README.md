# Emotiv Learn

Person 1 decision-policy baseline for the Emotiv Learn project.

## Contents

- `emotiv_learn/`: shared schemas and the linear contextual bandit `DecisionEngine`
- `tests/`: basic behavior checks for scoring and online updates
- `scripts/integration_driver.py`: stubbed end-to-end loop that shows how System 1 and System 2 fit together

## Quick Start

```bash
python3 -m pytest -q
```

## Integration Driver

Run a stubbed end-to-end loop across System 1 and lightweight System 2 placeholders:

```bash
python3 scripts/integration_driver.py
```

## End-to-End Smoke Flow

To run the full smoke-style flow across stub System 3, System 2, and the real System 1 engine:

```bash
python3 scripts/end_to_end_demo.py
python3 scripts/log_viewer.py
```

This writes:

- `artifacts/turn_logs.json`: canonical per-turn logs
- `artifacts/turn_log_viewer.html`: a lightweight viewer for inspecting the full handoff

The viewer shows, for each turn:

- System 3 raw observation
- System 2 state
- System 1 action scores and selected action
- System 2 interaction effect
- System 3 outcome
- System 2 reward event returned to System 1

## LLM Contract Checks

Run the prompt/reward contract without external API calls:

```bash
python3 scripts/llm_contract_dry_run.py
```

Run the same contract path with real OpenAI calls:

```bash
OPENAI_API_KEY=... python3 scripts/llm_contract_live_run.py \
  --topic "gradient descent" \
  --action-id worked_example
```

You can optionally set `OPENAI_MODEL`; otherwise the lightweight client uses its default model.

Run a short live LLM-backed policy-training loop without EEG features:

```bash
OPENAI_API_KEY=... python3 scripts/live_llm_training_loop.py \
  --topic "gradient descent" \
  --turns 3 \
  --user-id live_user_a
```

This loop builds state from recent interpreted LLM signals rather than EEG:

- confusion/comprehension/engagement/progress scores
- response type one-hot values
- student confidence
- turn progress and difficulty
- previous reward

The live loop now uses a hidden-knowledge student simulator:

- System 3 keeps private per-concept mastery, confidence, curiosity, attention, fatigue, engagement, style preferences, and misconceptions.
- System 3 evaluates each Tutor LLM message from the student's perspective, updates hidden knowledge, emits observable learner signals, and samples the next learner response type.
- System 1 only sees the fixed-length observable state vector built from emitted signals. It never receives hidden mastery or hidden preferences.
- `oracle_mastery_gain` is logged for simulator evaluation only. It is not included in the `RewardEvent` used to update the policy.
- The Student LLM only verbalizes the sampled response type; it does not choose whether the learner continues, clarifies, or branches.

Compare the hidden-knowledge simulator across learned and non-learned baselines:

```bash
python3 scripts/knowledge_policy_comparison.py --turns 100 --seed 17
```

This compares:

- `personalized`: shared generic weights plus user residuals
- `generic`: learned shared weights only
- `fixed_no_change`: no learned policy; always uses `no_change`
- `random`: no learned policy; samples actions uniformly

Run the live LLM comparison with a JSONL stream for the observability dashboard:

```bash
python3 scripts/live_policy_comparison.py \
  --turns 10 \
  --seed 17 \
  --output artifacts/live_policy_comparison_10turn.json \
  --events-output observability/public/live_policy_comparison_stream.jsonl
```

Launch the React observability dashboard:

```bash
cd observability
npm install
npm run dev
```

The dashboard polls `public/live_policy_comparison_stream.jsonl` while the experiment is running. You can also load a completed `artifacts/live_policy_comparison_*.json` file from the dashboard.

The driver demonstrates this contract flow:

1. A raw observation is converted into a fixed-length `State` by a stub `DemoStateBuilder`
2. `DecisionEngine.score_actions(...)` scores the canonical action bank
3. `DecisionEngine.select_action(...)` chooses one action
4. A stub `Outcome` is interpreted into semantic signals by `DemoOutcomeInterpreter`
5. `DemoRewardModel` computes a scalar reward and constructs a `RewardEvent`
6. `DecisionEngine.update(...)` learns from that `RewardEvent`

## Person 1 / Person 2 Handoff

The integration driver is meant to clarify responsibilities before the real System 2 implementation exists.

### Person 1 owns

- `DecisionEngine`
- action scoring
- action selection
- online policy update from `RewardEvent`
- optional persistence of generic weights and user residuals via SQLite

Person 1 should treat reward semantics as external. The engine consumes:

- `State`
- canonical `ACTION_BANK`
- `RewardEvent`

If you want weights to persist across runs, initialize the engine with a `db_path`:

```python
engine = DecisionEngine(feature_dim=15, db_path="artifacts/decision_engine.sqlite")
```

Useful stability knobs are also available:

```python
engine = DecisionEngine(
    feature_dim=15,
    db_path="artifacts/decision_engine.sqlite",
    reward_clip_abs=1.5,
    l2_weight_decay=0.001,
    update_clip_abs=0.1,
)
```

The engine also retains a bounded in-memory `update_history` with decision-time policy metadata and post-update error information for debugging.

### Person 2 owns

- building `State` from raw observation
- interpreting `Outcome` into structured learning signals
- computing scalar reward
- constructing `RewardEvent`

In the driver, these responsibilities are represented by:

- `DemoStateBuilder`
- `DemoOutcomeInterpreter`
- `DemoRewardModel`

### Practical integration rule

When Person 2 is ready, the stub classes in `scripts/integration_driver.py` can be replaced with real implementations as long as they preserve the same handoff:

```text
RawObservation -> State -> DecisionEngine -> Action
Outcome -> InterpretedOutcome -> RewardEvent -> DecisionEngine.update(...)
```

Person 1 should never infer reward directly from raw outcome text, and Person 2 should never modify policy weights directly.

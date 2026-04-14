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

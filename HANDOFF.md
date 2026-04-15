# Emotiv Learn Handoff

This branch contains the current Person 1/System 1 decision engine plus the new hidden-knowledge simulator and observability tooling for policy experiments.

## Current Architecture

The core loop is:

1. System 2 builds an observable `State` from prior emitted learner signals.
2. System 1 `DecisionEngine` scores the canonical action bank and selects one tutor action.
3. Tutor LLM generates the tutor message for that action.
4. System 3 `HiddenKnowledgeStudent` privately evaluates the tutor message, updates hidden concept mastery, emits observable signals, and samples the next learner response type.
5. Student LLM verbalizes the sampled response type.
6. Reward is computed from observable/interpreted signals.
7. System 1 updates from `RewardEvent`.

Important boundary: System 1 never sees hidden mastery, hidden preferences, or `oracle_mastery_gain`. Those are simulator-only truth used for evaluation.

## Important Files

- `emotiv_learn/decision_engine.py`: linear contextual bandit with generic weights and optional user residuals.
- `emotiv_learn/student_model.py`: hidden-knowledge student simulator.
- `emotiv_learn/live_training.py`: fixed-length observable state builder for live LLM experiments.
- `scripts/knowledge_policy_comparison.py`: cheap non-LLM baseline comparison across policy modes.
- `scripts/live_policy_comparison.py`: real Tutor LLM + Student LLM policy comparison with JSONL event streaming.
- `observability/`: React dashboard for viewing live JSONL streams or completed experiment artifacts.

## Run Checks

```bash
python3 -m pytest -q
cd observability
npm install
npm run build
```

## Run The Live Comparison

```bash
python3 scripts/live_policy_comparison.py \
  --turns 10 \
  --seed 17 \
  --output artifacts/live_policy_comparison_10turn.json \
  --events-output observability/public/live_policy_comparison_stream.jsonl
```

## Run The Dashboard

```bash
cd observability
npm run dev
```

Open the URL Vite prints. The dashboard should show `Emotiv Learn Observatory`.

## Latest Experiment Readout

The latest 10-turn live comparison showed:

- Personalized beats generic and fixed `no_change`.
- Random still beats personalized.
- Reward and oracle mastery gain are only weakly correlated.

Interpretation: the architecture is working, but the simulator/reward surface is still too forgiving to arbitrary action variety. Random wins because many actions currently produce useful novelty without enough mismatch penalty.

## Recommended Next Work

1. Add stronger mismatch penalties in `HiddenKnowledgeStudent`.
2. Make user preferences more decisive, especially for `example_builder` and `visual_scanner`.
3. Smooth checkpoint reward so correctness does not dominate all learning evidence.
4. Improve cold-start exploration so learned policies do not over-default to `no_change`.
5. Rerun `scripts/live_policy_comparison.py` and inspect reward vs `oracle_mastery_gain` in the dashboard.

# Emotiv Learn

This repo is the current working version of the Emotiv Learn policy-engine playground.

## What This Repo Is

This codebase is centered on a split-latent simulator for adaptive tutoring:

- `knowledge_state` models what the learner knows
- `neuro_state` models workload / fatigue / attention-like neurocognitive factors
- behavioral signals are emitted from both
- EEG-like signals are emitted primarily from `neuro_state`

The point of the simulator is not to reproduce real EEG perfectly. It is to test whether the decision engine benefits from an additional EEG-informed observation channel beyond behavior alone.

## Core Validation Question

Can the policy engine learn better tutoring actions when it gets:

1. behavior-only state
2. behavior + EEG-like state
3. behavior + EEG-like state + optional tutor-facing proxy features

## Important Architectural Assumption

The current repo does **not** rely on real EEG ingestion or old dataset-conditioned retrieval.

The EEG path is now:

- synthetic
- dynamic
- consistent with the simulator's hidden neuro state
- intended to validate the policy architecture, not physiological realism

## Main Components

- `emotiv_learn/decision_engine.py`
  Contextual bandit with generic + personalized residual weights.
- `emotiv_learn/student_model.py`
  Split-latent learner simulator.
- `emotiv_learn/eeg.py`
  Synthetic EEG emission layer.
- `emotiv_learn/live_training.py`
  State construction logic and ablation profiles.
- `emotiv_learn/reward_model.py`
  Deterministic reward functions from interpreted learner signals.
- `scripts/experiments/live_policy_comparison.py`
  Main live experiment runner.
- `scripts/experiments/run_live_state_ablation.py`
  Runs the three ablation profiles.
- `scripts/experiments/plot_state_ablation.py`
  Renders matplotlib plots from ablation outputs.

## How Another Agent Should Think About This Repo

If you are another LLM or coding agent:

- treat `SIMULATOR_ARCHITECTURE.md` as the design source of truth
- assume synthetic EEG only
- do not reintroduce the removed real-data ingestion path unless explicitly requested
- focus on:
  - reward design
  - action separation
  - state usefulness
  - ablation interpretation

## Most Useful Commands

Run the ablation:

```bash
python3 /Users/anish/PERSONAL/emotiv_learn/scripts/experiments/run_live_state_ablation.py \
  --turns 10 \
  --seed 17 \
  --model gpt-5.4-mini \
  --modes personalized,generic,fixed_no_change \
  --eeg-mode synthetic \
  --output-dir /Users/anish/PERSONAL/emotiv_learn/artifacts/state_ablation
```

Plot the ablation:

```bash
python3 /Users/anish/PERSONAL/emotiv_learn/scripts/experiments/plot_state_ablation.py \
  --input-dir /Users/anish/PERSONAL/emotiv_learn/artifacts/state_ablation \
  --output /Users/anish/PERSONAL/emotiv_learn/artifacts/state_ablation/ablation_summary.png
```

Run one live simulator workflow:

```bash
python3 /Users/anish/PERSONAL/emotiv_learn/scripts/experiments/live_llm_training_loop.py \
  --topic "gradient descent" \
  --user-id live_user_a \
  --turns 5 \
  --difficulty medium \
  --model gpt-5.4-mini \
  --db-path /Users/anish/PERSONAL/emotiv_learn/artifacts/live_llm_engine.sqlite \
  --output /Users/anish/PERSONAL/emotiv_learn/artifacts/live_llm_turns.json \
  --eeg-mode synthetic
```

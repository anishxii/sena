# Split-Latent Simulator Architecture

## Goal

Validate the decision engine under a multimodal simulator where behavior and EEG are distinct observation channels rather than duplicate readouts of the same scalar hidden state.

## Core Idea

The simulator now maintains two hidden substates:

1. `knowledge_state`
   - concept mastery
   - misconception strength
   - confidence
   - curiosity
   - pedagogical style preference
2. `neuro_state`
   - workload
   - fatigue
   - attention
   - vigilance
   - stress
   - engagement

Tutor actions update both substates. Behavior is emitted from both substates. EEG is emitted primarily from `neuro_state`, with only light coupling to knowledge mismatch through semantic friction and concept mastery.

## Why This Is Simpler

- We no longer claim a physiology-faithful EEG simulator.
- We only require EEG to be a dynamic, policy-relevant observation channel.
- The simulator now validates the policy architecture rather than exact biometric realism.

## Emission Model

### Behavior channel

Behavior observables are emitted from joint knowledge + neuro state:

- confusion
- comprehension
- engagement
- progress
- pace fast / slow
- checkpoint success probability
- response mode (`continue`, `clarify`, `branch`)

### EEG channel

EEG summary features are emitted primarily from neuro state:

- workload
- fatigue
- attention
- vigilance
- stress

The learned COG-BCI proxy regressor remains a secondary mapping from EEG summary features into higher-level experiment-aligned EEG proxies.

## Evaluation Contract

The ablation study should compare:

1. `behavior_only`
2. `current_eeg`
3. `tutor_proxy_eeg`

The key validation claim is:

> When EEG provides information not already present in behavioral observations, the policy engine should outperform a behavior-only baseline.

This validates the decision architecture, not full real-world physiological fidelity.

# Training Simulation Guide

## Overview

The training simulation orchestrates the full Emotiv Learn pipeline to generate training data for the RL personalization model. It simulates a complete learning session where:

1. An LLM generates a lesson plan on a given topic
2. Content is presented step-by-step to a simulated user (using real STEW EEG data)
3. The simulated user responds with EEG features, behavioral cues, and actions
4. Random content optimization actions are applied (placeholder for RL model)
5. All data is logged for RL training

## Quick Start

### Prerequisites

1. **STEW Dataset**: Place the STEW dataset in `../stew_dataset/` or specify path with `--stew-dir`
   - Should contain: `sub01_hi.txt`, `sub02_hi.txt`, ..., `ratings.txt`

2. **Environment Variables**: Create `../.env` file in parent directory with:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here  # for all LLM calls
   ```

3. **Dependencies**: Install required packages:
   ```bash
   pip install numpy scipy python-dotenv
   ```

### Run a Training Simulation

```bash
# Basic usage
python train_simulation.py --topic "derivatives and how to calculate them"

# Full options
python train_simulation.py \
    --topic "machine learning gradient descent" \
    --stew-dir ../stew_dataset \
    --num-steps 8 \
    --subject-id sub01 \
    --seed 42 \
    --output-dir ./training_logs
```

### Parameters

- `--topic` (required): Learning topic for the lesson
- `--stew-dir`: Path to STEW dataset directory (default: `../stew_dataset`)
- `--num-steps`: Number of lesson steps to generate (default: 10)
- `--subject-id`: Specific STEW subject to use, e.g., `sub01` (default: random)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output-dir`: Where to save training logs (default: `./training_logs`)

## Output

### Training Logs

Each simulation creates a JSON file in the output directory:
```
./training_logs/session_20260414_153022.json
```

The log contains:
- Session metadata (topic, subject, lesson plan)
- Per-step data:
  - Content text
  - EEG features (62-dim vector)
  - Behavioral cues (time, scroll rate, reread count)
  - Cognitive load score
  - User action (continue/clarify/branch)
  - User prompt (if any)
  - RL action applied
  - Content optimization action

### Example Output Structure

```json
{
  "session_id": "session_20260414_153022",
  "topic": "derivatives and how to calculate them",
  "subject_id": "sub01",
  "subject_rating": 6.0,
  "lesson_plan": [
    {"title": "Introduction to Derivatives", "outline": "..."},
    ...
  ],
  "steps": [
    {
      "step_number": 1,
      "step_title": "Introduction to Derivatives",
      "content": "...",
      "content_length": 342,
      "optimization_action_applied": "maintain",
      "eeg_features": [0.12, 0.34, ...],  // 62 values
      "behavioral_cues": {
        "time_on_chunk": 52.3,
        "scroll_rate": 0.823,
        "reread_count": 1
      },
      "cognitive_load": 0.456,
      "epochs_consumed": 2,
      "user_action": "continue",
      "user_prompt": null,
      "rl_action": "add_examples"
    },
    ...
  ]
}
```

## Analyzing Training Logs

Use `analyze_training_logs.py` to visualize and understand the training data:

### Single Session Analysis

```bash
python analyze_training_logs.py --log-file ./training_logs/session_20260414_153022.json
```

This prints:
- Summary statistics (avg cognitive load, time per step, etc.)
- User action distribution (continue/clarify/branch)
- RL action distribution
- Step-by-step breakdown
- Interesting moments (high cognitive load, clarifications, branches)

### Multi-Session Comparison

```bash
python analyze_training_logs.py --log-dir ./training_logs --compare
```

Analyzes all sessions in the directory and prints a comparison table.

### Export for RL Training

```bash
python analyze_training_logs.py \
    --log-file ./training_logs/session_20260414_153022.json \
    --export-rl ./rl_training_data.csv
```

Exports state transitions in CSV format:
```
state_eeg_1, ..., state_eeg_62, state_time, state_scroll, state_reread, action, reward, next_state_eeg_1, ...
```

Where:
- **State**: 62 EEG features + 3 behavioral cues = 65-dim vector
- **Action**: Index into `CONTENT_ACTIONS = ["simplify", "deepen", "add_examples", "use_analogy", "change_format", "maintain"]`
- **Reward**: Placeholder (currently `1 - cognitive_load`; should use prompt similarity + engagement)
- **Next State**: Same format as state

## Content Optimization Actions

The RL model chooses from 6 actions to modify content style:

| Action | Description |
|--------|-------------|
| `simplify` | Reduce complexity, simpler language, shorter sentences |
| `deepen` | Add technical depth, mathematical details, advanced concepts |
| `add_examples` | Include 2-3 concrete worked examples or case studies |
| `use_analogy` | Explain using analogies and metaphors |
| `change_format` | Switch presentation style (narrative ↔ bullets ↔ Q&A) |
| `maintain` | Keep current style (neutral action) |

Currently, actions are sampled randomly. Replace this with your RL model:

```python
# In train_simulation.py, line ~200:
# Replace:
next_action = rng.choice(CONTENT_ACTIONS)

# With:
next_action = rl_model.choose_action(
    eeg_features=user_response['eeg_features'],
    behavioral_cues=user_response['behavioral_cues']
)
```

## Integration with RL Model

### State Space (65 dimensions)

```python
state = np.concatenate([
    user_response['eeg_features'],           # 62 dims
    [user_response['behavioral_cues']['time_on_chunk']],
    [user_response['behavioral_cues']['scroll_rate']],
    [user_response['behavioral_cues']['reread_count']],
])
```

### Action Space (6 actions)

```python
CONTENT_ACTIONS = ["simplify", "deepen", "add_examples", "use_analogy", "change_format", "maintain"]
```

### Reward Signal

The reward should combine:
1. **Prompt similarity score** (from LLM User Interface module) — measures understanding
2. **Time on chunk** — engagement proxy
3. **Cognitive load** — avoid overload or boredom
4. **User action** — penalize "clarify" (confusion), reward "continue" or "branch" (flow/curiosity)

Example reward function:
```python
def compute_reward(user_response, prompt_similarity_score):
    load = user_response['cognitive_load']
    action = user_response['next_action']
    time = user_response['behavioral_cues']['time_on_chunk']

    # Understanding component (0-1)
    understanding = prompt_similarity_score

    # Engagement component: prefer moderate load (sweet spot ~0.4-0.6)
    engagement = 1.0 - abs(load - 0.5) * 2

    # Action penalty: clarify = -0.3, continue = 0, branch = +0.2
    action_bonus = {"continue": 0.0, "clarify": -0.3, "branch": 0.2}[action]

    # Time component: prefer reasonable engagement (60-90s)
    time_score = 1.0 if 60 <= time <= 90 else max(0, 1.0 - abs(time - 75) / 75)

    # Weighted combination
    reward = (
        0.5 * understanding +
        0.2 * engagement +
        0.2 * time_score +
        0.1 * action_bonus
    )

    return reward
```

## Example Workflow

```bash
# 1. Run training simulation for multiple topics
python train_simulation.py --topic "derivatives" --num-steps 8 --seed 1
python train_simulation.py --topic "neural networks" --num-steps 10 --seed 2
python train_simulation.py --topic "probability" --num-steps 6 --seed 3

# 2. Analyze all sessions
python analyze_training_logs.py --log-dir ./training_logs --compare

# 3. Export for RL training
python analyze_training_logs.py \
    --log-file ./training_logs/session_20260414_153022.json \
    --export-rl ./rl_data.csv

# 4. Train RL model (your friend's code)
python train_rl_model.py --data ./rl_data.csv

# 5. Replace random action selection with trained model
# Edit train_simulation.py to use rl_model.choose_action(...)
```

## Troubleshooting

### API Key Issues

```
[ERROR] OPENAI_API_KEY not found in environment
```

**Solution**: Create `../.env` file in parent directory with:
```
OPENAI_API_KEY=sk-...
```

### STEW Dataset Not Found

```
FileNotFoundError: No *_hi.txt files found in ../stew_dataset
```

**Solution**:
- Download STEW dataset
- Place `sub01_hi.txt`, ..., `sub36_hi.txt` in `../stew_dataset/`
- Add `ratings.txt` file
- Or specify correct path with `--stew-dir /path/to/stew`

### Out of Epochs

If simulation runs too long and exhausts EEG epochs:
- The agent automatically loops epochs by default (`loop_epochs=True`)
- Or it advances to the next subject (`loop_epochs=False`)
- No action needed — this is expected behavior

## Next Steps

1. **Run simulations** across diverse topics to build training dataset
2. **Integrate RL model** — replace random action selection with your friend's model
3. **Implement reward function** — add prompt similarity from LLM User Interface
4. **Train & evaluate** — use logged data to train contextual bandit
5. **Deploy** — connect to real LLM UI and simulated user for live testing

## File Reference

- [train_simulation.py](train_simulation.py) — Main simulation orchestrator
- [analyze_training_logs.py](analyze_training_logs.py) — Log analysis and visualization
- [simulated_user.py](simulated_user.py) — Simulated user agent

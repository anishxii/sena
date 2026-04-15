# Training Simulation Module

This folder contains everything needed to run training simulations for the Emotiv Learn RL personalization model.

## Files

- **[train_simulation.py](train_simulation.py)** - Main orchestration script
  - Generates lesson plans using OpenAI GPT-4
  - Feeds content to simulated user step-by-step
  - Collects training data for RL model
  - Currently uses random content optimization actions (placeholder for RL model)

- **[simulated_user.py](simulated_user.py)** - Simulated human learner agent
  - Replays real EEG data from STEW dataset
  - Produces 62-dim EEG feature vectors
  - Generates behavioral cues (time on chunk, scroll rate, reread count)
  - Samples user actions (continue/clarify/branch)
  - Creates realistic student prompts using GPT-4

- **[analyze_training_logs.py](analyze_training_logs.py)** - Log analysis tool
  - Visualize training session metrics
  - Compare multiple sessions
  - Export data to CSV for RL training

- **[example_run.sh](example_run.sh)** - Quick test script
  - Validates setup (API keys, STEW dataset)
  - Runs a short 5-step simulation

- **[TRAINING_SIMULATION.md](TRAINING_SIMULATION.md)** - Complete documentation
  - Detailed usage guide
  - Parameter reference
  - Integration instructions
  - Troubleshooting

## Quick Start

```bash
cd train

# Run a quick test (5 steps)
./example_run.sh

# Run a full simulation
python train_simulation.py \
    --topic "derivatives and how to calculate them" \
    --stew-dir ../stew_dataset \
    --num-steps 10

# Analyze the results
python analyze_training_logs.py --log-dir ./training_logs
```

## Requirements

1. **STEW Dataset**: Should be in `../stew_dataset/`
   - Contains `sub01_hi.txt`, `sub02_hi.txt`, ..., `ratings.txt`

2. **API Key**: Create `../.env` file with:
   ```
   OPENAI_API_KEY=sk-...
   ```

3. **Python packages**:
   ```bash
   pip install numpy scipy python-dotenv
   ```

## Output

Training logs are saved to `./training_logs/session_YYYYMMDD_HHMMSS.json`

Each log contains:
- Lesson plan
- Per-step content, EEG features (62-dim), behavioral cues, user actions
- RL actions applied (currently random)

## Integration with RL Model

To replace random actions with your trained RL model, edit [train_simulation.py:350](train_simulation.py#L350):

```python
# Replace:
next_action = rng.choice(CONTENT_ACTIONS)

# With:
next_action = rl_model.choose_action(
    eeg_features=user_response['eeg_features'],
    behavioral_cues=user_response['behavioral_cues']
)
```

State space: 65 dimensions (62 EEG + 3 behavioral)
Action space: 6 actions (simplify, deepen, add_examples, use_analogy, change_format, maintain)

See [TRAINING_SIMULATION.md](TRAINING_SIMULATION.md) for full integration guide.

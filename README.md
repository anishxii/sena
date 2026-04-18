# Sena

Sena is now organized around three active pieces:

- `System 1`: a reusable personalized decision engine
- `System 2`: a reusable cognitive middleware SDK
- `System 3`: a tutor demo / simulator built on top of Systems 1 and 2

The repo also includes a polished observability surface for replaying the cognitive pipeline and presentation artifacts for the hackathon story.

## Repo Map

```text
.
├── README.md                               # main contributor / run guide
├── CONTEXT_DUMP.md                         # temporary long-form project context for teammate or LLM handoff
├── artifacts/
│   └── replays/
│       └── sample_stream.jsonl             # canonical sample event stream for observability / replay
├── observability/                          # Sena dashboard + static presentation artifacts
│   ├── index.html                          # Vite entrypoint
│   ├── chat-comparison.html                # light-mode side-by-side tutor comparison artifact
│   ├── comparison.html                     # alternate static comparison artifact
│   ├── package.json                        # frontend tooling
│   ├── public/
│   │   ├── system3_comparison.json         # demo comparison data
│   │   └── system3_demo.json               # demo pipeline data
│   └── src/
│       ├── App.jsx                         # Sena observability application root
│       ├── ChatComparisonApp.jsx           # presentation chat comparison UI
│       ├── ComparisonApp.jsx               # secondary comparison UI
│       ├── components/                     # dashboard layout, canvas, panels, ui primitives
│       ├── config/                         # navigation + pipeline topology config
│       ├── replay/                         # replay view-model helpers
│       └── styles/                         # dashboard styling
├── systems/
│   ├── system1_decision/                   # contextual bandit engine
│   │   ├── engine.py                       # generic + personalized residual learner
│   │   └── schemas.py                      # canonical action bank + decision traces
│   ├── system2_sdk/                        # reusable middleware contract
│   │   ├── core/interfaces.py              # abstract app hooks
│   │   ├── core/runtime.py                 # canonical turn runner
│   │   ├── core/streaming.py               # stream event schema for observability
│   │   ├── core/types.py                   # shared SDK dataclasses
│   │   └── core/validation.py              # invariants and schema checks
│   └── system3b_tutor/                     # tutor-specific app / simulator (current package path)
│       ├── eeg.py                          # synthetic EEG observation layer
│       ├── live_training.py                # tutor state builder + feature profiles
│       ├── llm_contracts.py                # tutor / student / interpreter prompts
│       ├── reward_model.py                 # tutor reward shaping
│       ├── student_model.py                # split-latent learner simulator
│       ├── tutor_proxy.py                  # tutor-facing proxy features
│       └── runs/
│           ├── knowledge_policy_comparison.py  # offline policy comparison loop
│           ├── live_llm_training_loop.py       # single-user live loop
│           └── live_policy_comparison.py       # multi-policy live comparison
└── tests/
    ├── test_system1_engine.py              # System 1 sanity checks
    ├── test_system2_streaming.py           # System 2 stream contract checks
    └── test_system3b_student_model.py      # System 3 simulator checks
```

## What Each System Owns

### System 1

`systems/system1_decision` owns action scoring, action selection, online updates, and optional per-user residual personalization. It should not own reward semantics or application-specific state design.

### System 2

`systems/system2_sdk` is the reusable contract between raw observations, featurized state, interaction effects, interpreted outcomes, reward events, and replay events. Application authors should be able to import this package and define their own state/action/reward semantics on top of it.

### System 3

`systems/system3b_tutor` is the current application layer. It defines:

- a tutor-specific learner simulator
- tutor / student / interpreter prompt contracts
- a synthetic EEG observation layer
- runnable policy comparison loops

This is a demo / experimentation surface, not a neuroscience benchmark package.

## Quick Start

### Python checks

```bash
pytest -q tests/test_system1_engine.py tests/test_system2_streaming.py tests/test_system3b_student_model.py
```

### Run the offline tutor policy comparison

```bash
python3 -m systems.system3b_tutor.runs.knowledge_policy_comparison --turns 10 --seed 7
```

### Run the live tutor loop

This uses real OpenAI calls and expects `OPENAI_API_KEY` in a local `.env`.

```bash
python3 -m systems.system3b_tutor.runs.live_llm_training_loop \
  --topic "gradient descent" \
  --user-id learner_a \
  --turns 5 \
  --difficulty medium \
  --model gpt-4o-mini \
  --db-path artifacts/live_llm_engine.sqlite \
  --output artifacts/live_llm_turns.json \
  --eeg-mode synthetic
```

### Run the live multi-policy comparison

```bash
python3 -m systems.system3b_tutor.runs.live_policy_comparison \
  --turns 10 \
  --seed 17 \
  --model gpt-4o-mini \
  --output artifacts/live_policy_comparison.json \
  --events-output artifacts/live_policy_events.jsonl \
  --policy-modes personalized,generic,fixed_no_change,random \
  --state-profile current_eeg \
  --eeg-mode synthetic
```

### Launch observability

```bash
cd observability
npm install
npm run dev
```

Then open:

- `/` for the `Sena` observability dashboard
- `/chat-comparison.html` for the light-mode side-by-side tutor comparison artifact
- `/comparison.html` for the alternate comparison page

## Contribution Notes

- Keep `System 1`, `System 2`, and `System 3` separated conceptually.
- Put reusable contracts in `system2_sdk`, not in the tutor app.
- Put tutor-specific heuristics and prompts in `system3b_tutor`, not in the SDK.
- Keep observability consuming replay / stream primitives instead of inventing a second schema.
- Avoid reintroducing deleted neuroscience benchmark code into this main repo unless that work is explicitly being revived.

## Current Project Position

This repo is now optimized for:

- a clean hackathon handoff
- a coherent demo story
- a reusable middleware + observability narrative
- a lightweight tutor simulation surface for policy comparisons

It is not currently positioned as a real-data EEG validation repository.

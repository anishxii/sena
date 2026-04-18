System 2 SDK is the reusable cognitive middleware layer that sits between the
policy engine and any concrete application environment.

It owns:
- portable shared data types
- abstract interfaces applications must implement
- invariant validation
- a small canonical runtime helper for one closed-loop turn
- logging / replay primitives
- stream events for observability and dashboard replay

It does not own:
- tutor-specific state features
- tutor-specific action semantics
- one fixed reward formula for every application

Applications should import this package and define their own:
- `StateBuilder`
- `ActionRegistry`
- `InteractionModel`
- `OutcomeInterpreter`
- `RewardModel`

Observability guidance:
- serialize canonical SDK objects instead of inventing a second schema
- use `StreamEvent` for replay and dashboard rendering
- prefer `build_turn_stream_events(...)` from a complete `TurnLog`

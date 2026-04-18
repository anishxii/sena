System 2 SDK is the reusable contract and orchestration layer that sits between
the policy engine and any concrete application environment.

Core responsibilities:
- define portable turn-level data types
- define abstract interfaces applications must implement
- validate canonical invariants
- provide a small runtime helper for the shared closed-loop turn
- provide logging hooks for replay / observability

Non-responsibilities:
- defining a tutor action bank
- defining an EEG benchmark state schema
- hardcoding reward semantics for any one application

Applications such as the 3A EEG benchmark and the 3B tutor demo should import
this package and provide their own implementations of:
- StateBuilder
- ActionRegistry
- InteractionModel
- OutcomeInterpreter
- RewardModel

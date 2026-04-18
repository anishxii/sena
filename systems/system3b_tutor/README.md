System 3 is the tutor demo / simulator built on top of the reusable systems.

What lives here:
- split-latent learner simulator
- tutor / student / interpreter prompt contracts
- synthetic EEG-like observation layer
- tutor-facing proxy features
- live and offline policy comparison entrypoints

What this package is for:
- proving the policy engine can drive an adaptive application
- generating replayable sessions for observability
- giving the hackathon a concrete, understandable demo surface

What this package is not:
- a real neuroscience benchmark
- a production tutoring backend
- the owner of the generic middleware contract

Key entrypoints:
- `python3 -m systems.system3b_tutor.runs.knowledge_policy_comparison`
- `python3 -m systems.system3b_tutor.runs.live_llm_training_loop`
- `python3 -m systems.system3b_tutor.runs.live_policy_comparison`

Note:
- the current package path is still `systems.system3b_tutor`
- the product / architecture name is `System 3`

This is the temporary long-form context dump for the Sena project. It is intentionally verbose, uneven, and handoff-oriented. The goal is not polish. The goal is to preserve as much decision context as possible for another teammate or model.

Problem definition:

We want a generalizable personalization system that can choose the next best interaction strategy for a human user based on cognitive state, not just on application behavior alone. In the original framing this was an EEG-driven tutor, but over time the project got reframed into three distinct layers:

1. a reusable decision engine that learns action preferences online
2. a reusable middleware layer that translates between raw observations, state, interpreted outcomes, reward events, and replay streams
3. an application surface that proves the engine and middleware can actually drive a personalized experience

The relevant problem is meaningful because many adaptive systems personalize only from sparse behavior after the fact. Our broader thesis is that cognitive-state-aware middleware can let applications respond more intelligently and more transparently. For the hackathon, the product framing is: build the control plane for cognition-aware personalization, with tutoring as the concrete demo application.

Why this matters:

- societal / technological: many AI products treat every user the same way even when the user is overloaded, fatigued, confused, or ready for more challenge
- product: there is a gap between low-level biosignals and application-facing decisions; most teams do not have a clean way to use such signals in a feedback loop
- systems: even if the exact EEG interpretation changes by domain, a reusable state/reward/action middleware can stay constant

Important reframing:

We spent a large amount of time attempting to validate an EEG-specific scientific claim directly in code using public datasets and synthetic emitters. That work became messy and difficult to defend in the main product repo. The final repo deliberately removes the heavy real-data experimentation layer and focuses the main product on the clean systems architecture:

- System 1 = strong and reusable
- System 2 = generalizable SDK / middleware
- System 3 = tutor demo / simulator
- observability = polished replay / debugging / storytelling surface

The scientific EEG justification can still be discussed using literature and future validation plans, but the main repo no longer tries to be both product and full neuroscience benchmark at once.

Concept and research foundation:

The core concept is “cognitive middleware.” Instead of building one monolithic tutoring app, the project introduces a layer between observations and actions. That layer:

- receives raw observations from an application or environment
- builds a state representation
- allows a policy engine to score candidate actions
- emits interaction effects
- interprets outcomes
- computes reward
- logs everything in a canonical replay format

This makes the project interdisciplinary in a defensible way:

- machine learning / bandits: online personalized action selection
- software architecture: portable contracts and observability
- HCI / product: adaptive interaction strategies
- neuroscience framing: cognitive state signals as an upstream input channel

The strongest research-backed claim we can comfortably stand behind is not “we solved EEG tutoring scientifically.” It is closer to:

- internal cognitive state matters
- a middleware layer can expose that state to applications in a structured way
- personalization should happen on interpretable state/action/reward primitives, not in an opaque end-to-end blob

If asked about neuroscience rigor, the safest framing is:

- the repo’s main contribution is systems and middleware
- the EEG-specific scientific validation is a separate layer / future work area
- public literature supports the hypothesis that EEG contains predictive signal about internal cognitive state and future performance, but this repo’s main value is how such signals plug into applications

Proposed solution:

The solution is a reusable stack with four visible layers:

1. System 1 Decision Engine
   A linear contextual bandit with global weights and optional per-user residual weights. It scores a fixed action bank, chooses actions with optional epsilon exploration, and updates online from reward events.

2. System 2 Cognitive Middleware SDK
   The reusable contract for application builders. It defines data types like RawObservation, State, ActionScores, InteractionEffect, Outcome, InterpretedOutcome, RewardEvent, and TurnLog. It also defines abstract interfaces for StateBuilder, ActionRegistry, InteractionModel, OutcomeInterpreter, and RewardModel, plus canonical stream events for observability.

3. System 3 Tutor Demo / Simulator
   A tutor-specific package that defines:
   - a split-latent student simulator
   - tutor / student / interpreter prompt contracts
   - a synthetic EEG-like observation layer
   - offline and live policy comparison runners

4. Sena Observatory
   A frontend that visualizes the full cognitive pipeline:
   raw observation -> cognitive state -> policy decision -> observed outcome / reward
   It is designed to be product-like rather than notebook-like. The goal is to make the cognitive loop legible.

How it addresses the problem:

- the decision engine learns online instead of using one fixed tutoring style
- the SDK makes the architecture reusable for other applications
- the tutor demo shows the architecture in a relatable setting
- the observability layer makes the whole thing intelligible to builders, judges, and future users

Innovation and potential impact:

The novelty is not “we invented a new bandit.” The novelty is the combination:

- a cognition-aware middleware abstraction
- reusable schemas and runtime
- observability designed around cognitive pipeline nodes
- a demo application proving the middleware can drive adaptive behavior

Potential impact:

- other teams could import the SDK and define their own application-specific observations, actions, and rewards
- a future EEG headset integration would slot into raw observations rather than requiring the entire app to be rebuilt
- the observability and replay layer could become the developer surface for adaptive human-in-the-loop systems

Feasibility and validation:

What is validated in the current repo:

- the System 1 bandit can score, select, and update consistently
- the System 2 streaming / replay schema is coherent and testable
- the System 3 tutor simulator can produce state transitions, observable signals, and policy comparison runs
- the observability UI can replay the pipeline and serve as a clean hackathon artifact

What is not scientifically validated in the current repo:

- that the synthetic EEG layer is physiologically accurate
- that the tutor simulator proves real-world learning improvement
- that EEG improves tutoring outcomes in a live measured human study

The right honest claim is:

- we built the systems infrastructure for cognition-aware personalization
- we demonstrated it on a tutor application
- we designed it so richer biosignal inputs can be plugged in cleanly

Evidence we can talk about:

- internal tests for the decision engine and middleware contract
- the live tutor comparison loops
- the visual observability surface
- literature support for the broader hypothesis that internal neural state can provide useful information beyond purely explicit behavior

Real-world examples / competitor angle:

Most adaptive tutoring products personalize on explicit behavior only: correctness, latency, clickstream, completion. Neurotech tools often stay at the signal-processing layer and do not expose a reusable application middleware contract. Our positioning is that we sit in between:

- not just “brainwaves dashboard”
- not just “another tutoring agent”
- a reusable cognitive middleware / control plane that applications can build on

Build process:

The project originally started with a heavier tutor simulator and several attempts at public-dataset EEG validation. Over time this became too scattered and difficult to defend in one repo, so the repo was simplified and cleaned around the three-system architecture.

Final build choices:

- Python for Systems 1, 2, and 3
- React + Vite for observability
- lightweight test coverage with pytest
- JSONL-style replay thinking for instrumentation
- a clear boundary between reusable middleware and application-specific code

Tools / frameworks:

- Python dataclasses and simple package structure for backend logic
- sqlite support in the bandit for persistent user residuals
- React / Vite for the observability UI
- OpenAI API client wrapper for live tutor runs
- pytest for basic contract tests

Practicality:

Implementation feasibility is high for the systems side:

- System 1 is already compact and reusable
- System 2 is portable and extensible
- System 3 proves an application can be built on top
- Sena shows the experience of interacting with the middleware

Scalability:

- the decision engine is intentionally lightweight and online
- the SDK can support multiple applications if each defines its own builders / interpreters / reward models
- replay events create a path toward batch analysis, dashboards, and offline debugging

Usability:

- the observability layer is the key usability artifact for judges and builders
- the tutor demo is the key usability artifact for end-to-end storytelling

Final product:

The final output is not one monolithic app. It is a clean product stack:

- System 1: reusable policy engine
- System 2: reusable cognitive middleware SDK
- System 3: tutor application + simulator
- Sena: observability UI and presentation artifacts

The current presentation story should likely be:

1. define the problem: today’s adaptive apps personalize too late and too opaquely
2. introduce the solution: a reusable cognitive middleware layer
3. show the architecture: raw observation -> cognitive state -> policy decision -> action -> outcome -> reward
4. show a tutor example: two users, same explicit behavior, different cognitive rails / different best action
5. show the observability surface: Sena
6. show the generalizability: this is an SDK / control plane, not just a tutor hack

Detailed technical notes by system:

System 1:

- file area: `systems/system1_decision`
- core class: `DecisionEngine`
- model: linear contextual bandit
- supports:
  - generic weights
  - per-user residual weights
  - epsilon exploration
  - reward clipping
  - optional weight decay
  - optional sqlite persistence
- consumes `State` and `RewardEvent`
- emits `ActionScores` and selected `Action`

Why System 1 is strong:

- simple and understandable
- easy to explain mathematically
- aligns with hackathon scope
- small enough that the novelty is clearly not claimed to be the optimizer itself

System 2:

- file area: `systems/system2_sdk`
- purpose: define the reusable contract between applications and the policy layer
- important types:
  - `RawObservation`
  - `State`
  - `Action`
  - `ActionScores`
  - `InteractionEffect`
  - `Outcome`
  - `InterpretedOutcome`
  - `RewardEvent`
  - `TurnLog`
- important interfaces:
  - `StateBuilder`
  - `ActionRegistry`
  - `InteractionModel`
  - `OutcomeInterpreter`
  - `RewardModel`
- important runtime pieces:
  - `run_turn`
  - stream event builders
  - validators

Why System 2 matters:

- it is the “generalizable SDK” story
- it lets the tutor be only one application, not the whole product
- it gives the observability system one canonical schema to read from

System 3:

- file area: `systems/system3b_tutor`
- purpose: current demo application surface
- includes:
  - synthetic EEG observation layer
  - live tutor state builder
  - prompt contracts for tutor / student / interpreter
  - reward model
  - split-latent learner simulator
  - runnable policy comparison scripts

Important caveat:

The synthetic EEG layer is an application-side simulation tool. It is not the core scientific claim. It exists to keep the tutor pipeline alive and replayable.

Observability:

- file area: `observability`
- primary artifact: the Sena dashboard
- complementary artifact: light-mode side-by-side chat comparison

What the observability product should communicate:

- the system is not a black box
- observations turn into state
- state turns into policy decisions
- decisions turn into application effects
- outcomes and rewards are visible over time
- this is a platform layer, not just a one-off demo

What changed in the cleanup:

- removed old real-data neuroscience benchmark code from the main repo
- removed large local datasets and stale logs
- removed the old top-level `emotiv_learn` package in favor of `systems/system3b_tutor`
- moved runnable tutor scripts under `systems/system3b_tutor/runs`
- preserved Systems 1, 2, 3, observability, and minimal tests

To-do list after cleanup:

High priority:

- tighten `systems/system3b_tutor/README.md`
- ensure live tutor runners still execute cleanly from the new package path
- make sure observability data sources align with the final replay schema
- run pytest on the cleaned minimal suite
- verify the Vite dashboard still launches after cleanup

Product / presentation:

- finalize the 7-minute pitch arc around middleware + observability + tutor demo
- produce a polished slide using the chat comparison artifact
- produce a polished slide showing the Sena topology dashboard
- explain clearly why the tutor is one example of a broader SDK

Technical next steps if time remained:

- create a concrete System 3 implementation of the abstract System 2 interfaces rather than parallel package logic
- emit canonical System 2 `TurnLog` objects directly from the tutor runners
- unify tutor replay generation with `StreamEvent`
- improve reward shaping and comparison instrumentation
- add more explicit application adapters to prove generality beyond tutoring

What to tell a teammate plainly:

The repo is now intentionally narrower and cleaner. It is not trying to prove every neuroscience claim in code. It is trying to present a strong architecture:

- a real reusable decision engine
- a real reusable middleware SDK
- a coherent application demo
- a beautiful observability story

That is the core product.

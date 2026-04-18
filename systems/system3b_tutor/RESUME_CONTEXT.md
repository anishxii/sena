# System 3B Resume Context

This note captures the decisions we made while narrowing the project scope so we can return to the tutor demo later without re-litigating architecture.

## Why 3B Was Simplified

We decided to stop treating the tutor simulator as scientific validation. The old end-to-end simulator mixed together too many claims:

- the personalization engine works,
- the tutor application works,
- the hidden learner state is realistic,
- the EEG emitter is realistic,
- and the whole loop reflects real cognition.

That bundled story was too fragile. A judge could challenge any one heuristic link and undermine the whole presentation.

Instead, we now separate:

- **System 3A** as the scientific benchmark:
  real EEG + task/behavior data used to show EEG adds value to state estimation.
- **System 3B** as the product/demo layer:
  a tutor application that shows the personalization engine and SDK working in an adaptive application.

## What 3B Should Demonstrate

System 3B is not responsible for proving that EEG improves learning. That claim belongs to 3A.

System 3B should show three things:

1. a concrete tutor application built on top of the SDK contracts,
2. visible action adaptation across a short interaction,
3. longer-run simulated comparisons where personalized policy beats generic or no-policy for different learner profiles.

## Current 3B Scope

We agreed to split 3B into two layers:

### 3B1: Demo Surface

A short, human-readable demo focused on presentation value.

Characteristics:

- likely 3 turns,
- highly controlled,
- mostly deterministic,
- shows how different learner profiles trigger different action choices,
- visually demonstrates that the application layer is using the policy output.

This is the thing judges can watch and understand quickly.

### 3B2: Lightweight Product Simulator

A simple environment used only to compare policies over longer rollouts.

Characteristics:

- interpretable learner profiles,
- simple hidden learner state,
- explicit transitions,
- simple reward semantics,
- enough structure to compare personalized vs generic vs no-policy,
- not presented as neuroscience validation.

This replaces the earlier over-ambitious simulator. It is acceptable for 3B2 to be synthetic as long as we are honest that it is a product/demo environment rather than evidence for EEG science.

## What We Are Not Doing

We are not currently building:

- a full real-user production tutor,
- a scientifically validated tutor reactivity model,
- a claim that the 3B simulator itself proves neural realism,
- or a complete live EEG-reactive tutoring environment.

That means some earlier work, such as the richer knowledge-graph-centric tutor simulation, is no longer core infrastructure. Parts of it may still be reused for content organization or scripted examples, but it should not drive the scientific argument.

## Relationship To Systems 1 and 2

- **System 1** remains the core decision engine and is treated as the strongest finished component.
- **System 2** is now the general SDK/runtime/middleware layer that applications plug into.
- **System 3B** should be built as an application on top of System 2, not as a one-off custom loop.

The right mental model is:

- 3A proves EEG-informed state estimation is valuable,
- 1 + 2 provide the reusable adaptive decision stack,
- 3B shows that stack inside a tutor application.

## Presentation Guidance

When presenting 3B, we should say:

- this is a demo application built on our personalization SDK,
- the demo simulator is for product behavior and policy comparison,
- the neuroscience evidence comes from the separate 3A benchmark,
- and the architecture is intentionally modular so the same engine could support other adaptive applications later.

## Next Time We Return To 3B

When we resume 3B, the first tasks should be:

1. define the exact 3B1 three-turn demo script,
2. define the minimal learner profiles and state variables for 3B2,
3. wire 3B onto System 2 interfaces instead of custom ad hoc loops,
4. only then decide what older simulator pieces are worth reusing.

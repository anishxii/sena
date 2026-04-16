from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConceptSpec:
    concept_id: str
    label: str
    canonical_claim: str
    prerequisites: tuple[str, ...]
    checkpoint_prompt: str


@dataclass(frozen=True)
class TopicScenario:
    topic_id: str
    title: str
    ordered_concepts: tuple[ConceptSpec, ...]
    initial_knowledge_base: tuple[str, ...]
    goal_knowledge_base: tuple[str, ...]

    @property
    def concept_ids(self) -> list[str]:
        return [concept.concept_id for concept in self.ordered_concepts]

    def concept(self, concept_id: str) -> ConceptSpec:
        for concept in self.ordered_concepts:
            if concept.concept_id == concept_id:
                return concept
        raise KeyError(concept_id)


BACKPROP_SCENARIO = TopicScenario(
    topic_id="backpropagation_gradient_descent",
    title="Backpropagation and Gradient Descent",
    ordered_concepts=(
        ConceptSpec(
            concept_id="neural_networks",
            label="neural networks",
            canonical_claim="Neural networks are trainable machine-learning models built from layers of weighted computations.",
            prerequisites=(),
            checkpoint_prompt="What makes a neural network trainable?",
        ),
        ConceptSpec(
            concept_id="loss_function",
            label="loss function",
            canonical_claim="A loss function measures how far the model's predictions are from the desired outputs.",
            prerequisites=("neural_networks",),
            checkpoint_prompt="What does the loss function tell us during training?",
        ),
        ConceptSpec(
            concept_id="gradient_descent",
            label="gradient descent",
            canonical_claim="Gradient descent updates parameters by stepping in the direction that reduces the loss.",
            prerequisites=("loss_function",),
            checkpoint_prompt="How does gradient descent use the loss to update parameters?",
        ),
        ConceptSpec(
            concept_id="gradient",
            label="gradient",
            canonical_claim="The gradient shows how each parameter change will affect the loss locally.",
            prerequisites=("loss_function", "gradient_descent"),
            checkpoint_prompt="What information does the gradient provide during optimization?",
        ),
        ConceptSpec(
            concept_id="chain_rule",
            label="chain rule",
            canonical_claim="The chain rule lets us decompose how upstream computations affect downstream outputs.",
            prerequisites=("gradient",),
            checkpoint_prompt="Why is the chain rule useful inside layered models?",
        ),
        ConceptSpec(
            concept_id="backpropagation",
            label="backpropagation",
            canonical_claim="Backpropagation applies the chain rule through the network to compute gradients efficiently for every weight.",
            prerequisites=("gradient", "chain_rule"),
            checkpoint_prompt="How does backpropagation use the chain rule to train a neural network?",
        ),
    ),
    initial_knowledge_base=(
        "Neural networks are an essential tool in machine learning.",
    ),
    goal_knowledge_base=(
        "A loss function measures how far the model's predictions are from the desired outputs.",
        "Gradient descent updates parameters by stepping in the direction that reduces the loss.",
        "The gradient shows how each parameter change will affect the loss locally.",
        "The chain rule lets us decompose how upstream computations affect downstream outputs.",
        "Backpropagation applies the chain rule through the network to compute gradients efficiently for every weight.",
    ),
)

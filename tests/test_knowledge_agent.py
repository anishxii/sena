from emotiv_learn.knowledge_agent import KnowledgeAgent, KnowledgeAgentProfile
from emotiv_learn.knowledge_scenarios import BACKPROP_SCENARIO


def test_knowledge_agent_emits_cognitive_appraisal() -> None:
    agent = KnowledgeAgent(
        scenario=BACKPROP_SCENARIO,
        profile=KnowledgeAgentProfile(user_id="test_user", initial_knowledge_level=0.45),
        seed=3,
    )
    concept_id = "gradient_descent"
    agent.state = type(agent.state)(
        knowledge_base=agent.state.knowledge_base,
        concept_mastery={**agent.state.concept_mastery, "loss_function": 0.72},
        confusion=agent.state.confusion,
        confidence=agent.state.confidence,
        curiosity=agent.state.curiosity,
        fatigue=agent.state.fatigue,
        engagement=agent.state.engagement,
        attention=agent.state.attention,
        current_concept_index=2,
        current_concept_steps=0,
        steps_taken=0,
    )

    turn = agent.consume_tutor_step(
        concept_id=concept_id,
        tutor_message=(
            "Step 1: calculate how the loss changes. "
            "Step 2: move the parameter in the direction that reduces loss. "
            "For example, if the gradient is negative, increasing the parameter can reduce the loss. "
            "Can you explain why this update moves toward a lower loss?"
        ),
        action_id="worked_example",
        checkpoint_expected=True,
    )

    assert "intrinsic_load" in turn.cognitive_appraisal
    assert "extraneous_load" in turn.cognitive_appraisal
    assert "germane_support" in turn.cognitive_appraisal
    assert "technical_density" in turn.instructional_signals
    assert turn.state_after.concept_mastery[concept_id] >= turn.state_before.concept_mastery[concept_id]

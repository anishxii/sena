export function buildDemoData() {
  return {
    sourceLabel: "embedded_demo",
    snapshots: [
      {
        eventId: "demo_001",
        eventLabel: "event 1",
        applicationType: "tutor",
        policyMode: "behavior_only_baseline",
        userId: "learner_a",
        contextSummary: "same outward behavior, higher latent overload",
        selectedAction: "highlight_key_points",
        reward: 0.18,
        actionScores: {
          highlight_key_points: 0.54,
          worked_example: 0.52,
          deepen: 0.34,
          simplify: 0.31,
        },
        rawObservation: {
          eeg: {
            theta_alpha_ratio: 1.82,
            workload_estimate: 0.74,
            lapse_rate: 0.58,
          },
          behavior: {
            followup_type: "continue",
            confidence: 0.63,
            checkpoint_correct: 1,
            progress_signal: 0.62,
          },
          context: {
            application: "tutor",
            topic: "gradient_descent",
            difficulty: "medium",
            event_index: 1,
          },
        },
        cognitiveState: {
          overload_risk: 0.74,
          future_lapse_risk: 0.58,
          attention_stability: 0.42,
          engagement: 0.61,
          confidence: 0.63,
          needs_structure: 0.49,
          readiness_for_depth: 0.44,
        },
        rationale:
          "Behavior-only baseline sees acceptable explicit progress, so it settles on a mild structural intervention.",
        applicationEffect: {
          effect_type: "content_reformat",
          summary:
            "Key points:\n- Gradient points uphill.\n- Gradient descent moves opposite the gradient.\n- Learning rate controls step size.",
        },
        outcome: {
          explicit_feedback: {
            response_type: "continue",
            student_message: "Okay, that makes sense.",
            checkpoint_answer: "A",
          },
          reward_signals: {
            reward: 0.18,
            oracle_mastery_gain: 0.05,
            comprehension_score: 0.56,
            confusion_score: 0.28,
            progress_signal: 0.62,
          },
        },
      },
      {
        eventId: "demo_002",
        eventLabel: "event 1",
        applicationType: "tutor",
        policyMode: "personalized",
        userId: "learner_a",
        contextSummary: "same outward behavior, higher latent overload",
        selectedAction: "worked_example",
        reward: 0.42,
        actionScores: {
          worked_example: 0.77,
          highlight_key_points: 0.49,
          simplify: 0.46,
          deepen: 0.21,
        },
        rawObservation: {
          eeg: {
            theta_alpha_ratio: 1.82,
            workload_estimate: 0.74,
            lapse_rate: 0.58,
          },
          behavior: {
            followup_type: "continue",
            confidence: 0.63,
            checkpoint_correct: 1,
            progress_signal: 0.62,
          },
          context: {
            application: "tutor",
            topic: "gradient_descent",
            difficulty: "medium",
            event_index: 1,
          },
        },
        cognitiveState: {
          overload_risk: 0.74,
          future_lapse_risk: 0.58,
          attention_stability: 0.42,
          engagement: 0.61,
          confidence: 0.63,
          needs_structure: 0.73,
          readiness_for_depth: 0.39,
        },
        rationale:
          "Personalized policy detects elevated overload risk despite acceptable outward behavior and shifts toward an example-first intervention.",
        applicationEffect: {
          effect_type: "worked_example",
          summary:
            "Example: if the gradient at x = 4 is +2 and the learning rate is 0.1, gradient descent updates x to 3.8. The model takes a small step downhill instead of guessing the whole path at once.",
        },
        outcome: {
          explicit_feedback: {
            response_type: "continue",
            student_message: "That example helps a lot more.",
            checkpoint_answer: "A",
          },
          reward_signals: {
            reward: 0.42,
            oracle_mastery_gain: 0.12,
            comprehension_score: 0.73,
            confusion_score: 0.16,
            progress_signal: 0.79,
          },
        },
      },
      {
        eventId: "demo_003",
        eventLabel: "event 1",
        applicationType: "tutor",
        policyMode: "personalized",
        userId: "learner_b",
        contextSummary: "same outward behavior, stable attention",
        selectedAction: "deepen",
        reward: 0.39,
        actionScores: {
          deepen: 0.74,
          highlight_key_points: 0.42,
          worked_example: 0.28,
          simplify: 0.18,
        },
        rawObservation: {
          eeg: {
            theta_alpha_ratio: 0.94,
            workload_estimate: 0.33,
            lapse_rate: 0.19,
          },
          behavior: {
            followup_type: "continue",
            confidence: 0.64,
            checkpoint_correct: 1,
            progress_signal: 0.64,
          },
          context: {
            application: "tutor",
            topic: "gradient_descent",
            difficulty: "medium",
            event_index: 1,
          },
        },
        cognitiveState: {
          overload_risk: 0.33,
          future_lapse_risk: 0.19,
          attention_stability: 0.77,
          engagement: 0.67,
          confidence: 0.64,
          needs_structure: 0.31,
          readiness_for_depth: 0.81,
        },
        rationale:
          "With stable attention and low overload risk, the personalized policy increases conceptual depth rather than adding more scaffolding.",
        applicationEffect: {
          effect_type: "depth_increase",
          summary:
            "Gradient descent follows local slope information, but the learning rate also controls optimization stability. Too large a step can overshoot the minimum even when the direction is correct.",
        },
        outcome: {
          explicit_feedback: {
            response_type: "branch",
            student_message: "Could this connect to why training sometimes oscillates?",
            checkpoint_answer: "A",
          },
          reward_signals: {
            reward: 0.39,
            oracle_mastery_gain: 0.11,
            comprehension_score: 0.76,
            confusion_score: 0.12,
            progress_signal: 0.72,
          },
        },
      },
    ],
  };
}

export function uniqueValues(values) {
  return [...new Set(values.filter(Boolean))];
}

export function formatNumber(value) {
  if (value == null || Number.isNaN(Number(value))) return "n/a";
  return Number(value).toFixed(3);
}

export function formatValue(value) {
  if (value == null) return "n/a";
  if (typeof value === "number") return formatNumber(value);
  return String(value);
}

export function countLeafValues(groups) {
  return Object.values(groups || {}).reduce((count, payload) => count + Object.keys(payload || {}).length, 0);
}

export function flattenGroups(groups) {
  const rows = [];
  Object.entries(groups || {}).forEach(([groupName, payload]) => {
    Object.entries(payload || {}).forEach(([key, value]) => rows.push([`${groupName}.${key}`, value]));
  });
  return rows;
}

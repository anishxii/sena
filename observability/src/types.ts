export type PolicyMode = 'personalized' | 'generic' | 'fixed_no_change' | 'random';

export type ExperimentEvent = {
  event_id: string;
  timestamp: string;
  event_type: string;
  payload: Record<string, unknown>;
};

export type TurnCompletedPayload = {
  policy_mode: PolicyMode;
  turn_index: number;
  user_id: string;
  concept_id: string;
  checkpoint_expected: boolean;
  action_id: string;
  tutor_message: string;
  student_response: Record<string, unknown>;
  interpreted: Record<string, number | string | boolean | null | Record<string, unknown>>;
  reward: number;
  oracle_mastery_gain: number;
  response_type: string;
  checkpoint_correct: boolean | null;
  student_transition: {
    hidden_state_before: Record<string, unknown>;
    hidden_state_after: Record<string, unknown>;
    evaluation: Record<string, number>;
    observable_signals: Record<string, unknown>;
    response_type_probs: Record<string, number>;
    sampled_response_type: string;
    oracle_mastery_gain: number;
  };
  update_trace?: Record<string, unknown> | null;
};

export type ExperimentArtifact = {
  turns: number;
  seed: number;
  model: string | null;
  results: Record<string, { users: UserSummary[]; turn_logs: TurnCompletedPayload[] }>;
  summary: Record<string, PolicySummary>;
};

export type UserSummary = {
  user_id: string;
  average_reward: number;
  total_oracle_mastery_gain: number;
  average_oracle_mastery_gain: number;
  checkpoint_accuracy: number;
  followups: Record<string, number>;
  action_counts: Record<string, number>;
};

export type PolicySummary = {
  average_reward: number;
  total_oracle_mastery_gain: number;
  average_oracle_mastery_gain: number;
  checkpoint_accuracy: number;
};

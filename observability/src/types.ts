export type PolicyMode = 'personalized' | 'generic' | 'fixed_no_change' | 'random';

export type ExperimentEvent = {
  event_id?: string;
  timestamp: string;
  event_type: string;
  payload: Record<string, unknown>;
};

export type LegacyTurnCompletedPayload = {
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

export type KnowledgeEvaluation = {
  goal_coverage_score: number;
  knowledge_quality_score: number;
  missing_goal_claims: string[];
  achieved_goal_claims: string[];
  goal_reached: boolean;
};

export type KnowledgeStepPayload = {
  policy_name: 'generic' | 'personalized';
  user_id: string;
  step_index: number;
  concept_id: string;
  checkpoint_expected: boolean;
  action_id: string;
  tutor_message: string;
  knowledge_turn: {
    concept_id: string;
    tutor_message: string;
    action_id: string;
    checkpoint_expected: boolean;
    state_before: Record<string, unknown>;
    state_after: {
      knowledge_base?: string[];
      concept_mastery?: Record<string, number>;
      confusion?: number;
      confidence?: number;
      curiosity?: number;
      fatigue?: number;
      engagement?: number;
      attention?: number;
      current_concept_index?: number;
      steps_taken?: number;
    };
    response_type: string;
    reprompt: string | null;
    self_reported_confidence: number;
    checkpoint_answer: string | null;
    checkpoint_correct: boolean | null;
    progress_signal: number;
  };
  evaluation: KnowledgeEvaluation;
  policy_metadata?: {
    action_id?: string;
    interpreted?: Record<string, unknown>;
    reward?: number;
    eeg?: {
      feature_names: string[];
      features: number[];
      metadata: Record<string, unknown>;
    };
  };
  knowledge_base?: string[];
  goal_coverage: number;
  knowledge_quality: number;
  goal_reached: boolean;
  response_type: string;
};

export type LegacyExperimentArtifact = {
  turns: number;
  seed: number;
  model: string | null;
  results: Record<string, { users: LegacyUserSummary[]; turn_logs: LegacyTurnCompletedPayload[] }>;
  summary: Record<string, LegacyPolicySummary>;
};

export type KnowledgeExperimentArtifact = {
  scenario: string;
  max_steps: number;
  seed: number;
  model: string | null;
  eeg_mode: string;
  results: Record<string, { users: KnowledgeUserResult[]; summary: KnowledgePolicySummary }>;
  comparison: Record<string, number | null>;
};

export type ExperimentArtifact = LegacyExperimentArtifact | KnowledgeExperimentArtifact;

export type LegacyUserSummary = {
  user_id: string;
  average_reward: number;
  total_oracle_mastery_gain: number;
  average_oracle_mastery_gain: number;
  checkpoint_accuracy: number;
  followups: Record<string, number>;
  action_counts: Record<string, number>;
};

export type LegacyPolicySummary = {
  average_reward: number;
  total_oracle_mastery_gain: number;
  average_oracle_mastery_gain: number;
  checkpoint_accuracy: number;
};

export type KnowledgeUserResult = {
  user_id: string;
  subject_id: string;
  steps_taken: number;
  steps_to_goal: number | null;
  goal_reached_within_budget: boolean;
  initial_knowledge_base: string[];
  final_knowledge_base: string[];
  final_state: Record<string, unknown>;
  final_evaluation: KnowledgeEvaluation;
  step_logs: KnowledgeStepPayload[];
};

export type KnowledgePolicySummary = {
  users_run: number;
  goals_reached: number;
  avg_final_goal_coverage: number;
  avg_final_knowledge_quality: number;
  avg_steps_taken: number;
  avg_steps_to_goal: number | null;
};

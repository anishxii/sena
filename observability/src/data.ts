import type {
  ExperimentArtifact,
  ExperimentEvent,
  KnowledgeExperimentArtifact,
  KnowledgePolicySummary,
  KnowledgeStepPayload,
  LegacyExperimentArtifact,
  LegacyPolicySummary,
  LegacyTurnCompletedPayload,
  PolicyMode,
} from './types';

export const legacyPolicyOrder: PolicyMode[] = ['personalized', 'generic', 'fixed_no_change', 'random'];
export const knowledgePolicyOrder: Array<'personalized' | 'generic'> = ['personalized', 'generic'];

export function parseJsonl(raw: string): ExperimentEvent[] {
  return raw
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as ExperimentEvent);
}

export function detectDatasetKind(
  events: ExperimentEvent[],
  artifact: ExperimentArtifact | null,
): 'knowledge' | 'legacy' {
  if (artifact && 'scenario' in artifact) return 'knowledge';
  if (events.some((event) => event.event_type.startsWith('knowledge_'))) return 'knowledge';
  return 'legacy';
}

export function legacyTurnsFromEvents(events: ExperimentEvent[]): LegacyTurnCompletedPayload[] {
  return events
    .filter((event) => event.event_type === 'turn_completed')
    .map((event) => event.payload as unknown as LegacyTurnCompletedPayload);
}

export function legacyTurnsFromArtifact(artifact: LegacyExperimentArtifact): LegacyTurnCompletedPayload[] {
  return Object.values(artifact.results).flatMap((policy) => policy.turn_logs ?? []);
}

export function knowledgeStepsFromEvents(events: ExperimentEvent[]): KnowledgeStepPayload[] {
  return events
    .filter((event) => event.event_type === 'knowledge_step_completed')
    .map((event) => event.payload as unknown as KnowledgeStepPayload);
}

export function knowledgeStepsFromArtifact(artifact: KnowledgeExperimentArtifact): KnowledgeStepPayload[] {
  return Object.values(artifact.results).flatMap((policy) =>
    policy.users.flatMap((user) => user.step_logs ?? []),
  );
}

export function summarizeLegacyTurns(turns: LegacyTurnCompletedPayload[]): Record<string, LegacyPolicySummary> {
  const grouped = new Map<string, LegacyTurnCompletedPayload[]>();
  for (const turn of turns) {
    const rows = grouped.get(turn.policy_mode) ?? [];
    rows.push(turn);
    grouped.set(turn.policy_mode, rows);
  }

  const summary: Record<string, LegacyPolicySummary> = {};
  for (const [policyMode, rows] of grouped) {
    const checkpointRows = rows.filter((row) => row.checkpoint_correct !== null);
    summary[policyMode] = {
      average_reward: mean(rows.map((row) => row.reward)),
      total_oracle_mastery_gain: sum(rows.map((row) => row.oracle_mastery_gain)),
      average_oracle_mastery_gain: mean(rows.map((row) => row.oracle_mastery_gain)),
      checkpoint_accuracy:
        checkpointRows.length === 0
          ? 0
          : checkpointRows.filter((row) => row.checkpoint_correct === true).length / checkpointRows.length,
    };
  }
  return summary;
}

export function summarizeKnowledgeSteps(steps: KnowledgeStepPayload[]): Record<string, KnowledgePolicySummary> {
  const grouped = new Map<string, KnowledgeStepPayload[]>();
  for (const step of steps) {
    const rows = grouped.get(step.policy_name) ?? [];
    rows.push(step);
    grouped.set(step.policy_name, rows);
  }

  const summary: Record<string, KnowledgePolicySummary> = {};
  for (const [policyName, rows] of grouped) {
    const userIds = Array.from(new Set(rows.map((row) => row.user_id)));
    const lastByUser = new Map<string, KnowledgeStepPayload>();
    for (const row of rows) {
      const current = lastByUser.get(row.user_id);
      if (!current || row.step_index >= current.step_index) {
        lastByUser.set(row.user_id, row);
      }
    }
    const finals = Array.from(lastByUser.values());
    const stepsToGoalValues = finals
      .filter((row) => row.goal_reached)
      .map((row) => row.step_index);

    summary[policyName] = {
      users_run: userIds.length,
      goals_reached: finals.filter((row) => row.goal_reached).length,
      avg_final_goal_coverage: mean(finals.map((row) => row.goal_coverage)),
      avg_final_knowledge_quality: mean(finals.map((row) => row.knowledge_quality)),
      avg_steps_taken: mean(finals.map((row) => row.step_index)),
      avg_steps_to_goal: stepsToGoalValues.length > 0 ? mean(stepsToGoalValues) : null,
    };
  }
  return summary;
}

export function legacyPolicyCounts(
  turns: LegacyTurnCompletedPayload[],
  key: 'action_id' | 'response_type',
): Record<string, Record<string, number>> {
  const counts: Record<string, Record<string, number>> = {};
  for (const turn of turns) {
    counts[turn.policy_mode] ??= {};
    const value = String(turn[key]);
    counts[turn.policy_mode][value] = (counts[turn.policy_mode][value] ?? 0) + 1;
  }
  return counts;
}

export function knowledgePolicyCounts(
  steps: KnowledgeStepPayload[],
  key: 'action_id' | 'response_type' | 'concept_id',
): Record<string, Record<string, number>> {
  const counts: Record<string, Record<string, number>> = {};
  for (const step of steps) {
    counts[step.policy_name] ??= {};
    const value = String(step[key]);
    counts[step.policy_name][value] = (counts[step.policy_name][value] ?? 0) + 1;
  }
  return counts;
}

export function filterLegacyTurns(turns: LegacyTurnCompletedPayload[], userId: string | null): LegacyTurnCompletedPayload[] {
  if (!userId) return turns;
  return turns.filter((turn) => turn.user_id === userId);
}

export function filterKnowledgeSteps(steps: KnowledgeStepPayload[], userId: string | null): KnowledgeStepPayload[] {
  if (!userId) return steps;
  return steps.filter((step) => step.user_id === userId);
}

export function uniqueLegacyUsers(turns: LegacyTurnCompletedPayload[]): string[] {
  return Array.from(new Set(turns.map((turn) => String(turn.user_id)))).sort();
}

export function uniqueKnowledgeUsers(steps: KnowledgeStepPayload[]): string[] {
  return Array.from(new Set(steps.map((step) => String(step.user_id)))).sort();
}

export function formatNumber(value: number | null | undefined, digits = 3): string {
  if (value === undefined || value === null || Number.isNaN(value)) return '—';
  return value.toFixed(digits);
}

export function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return sum(values) / values.length;
}

export function sum(values: number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

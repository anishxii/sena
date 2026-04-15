import type { ExperimentArtifact, ExperimentEvent, PolicyMode, PolicySummary, TurnCompletedPayload } from './types';

export const policyOrder: PolicyMode[] = ['personalized', 'generic', 'fixed_no_change', 'random'];

export function parseJsonl(raw: string): ExperimentEvent[] {
  return raw
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as ExperimentEvent);
}

export function turnsFromEvents(events: ExperimentEvent[]): TurnCompletedPayload[] {
  return events
    .filter((event) => event.event_type === 'turn_completed')
    .map((event) => event.payload as unknown as TurnCompletedPayload);
}

export function turnsFromArtifact(artifact: ExperimentArtifact): TurnCompletedPayload[] {
  return Object.values(artifact.results).flatMap((policy) => policy.turn_logs ?? []);
}

export function summarizeTurns(turns: TurnCompletedPayload[]): Record<string, PolicySummary> {
  const grouped = new Map<string, TurnCompletedPayload[]>();
  for (const turn of turns) {
    const rows = grouped.get(turn.policy_mode) ?? [];
    rows.push(turn);
    grouped.set(turn.policy_mode, rows);
  }

  const summary: Record<string, PolicySummary> = {};
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

export function policyCounts(turns: TurnCompletedPayload[], key: 'action_id' | 'response_type'): Record<string, Record<string, number>> {
  const counts: Record<string, Record<string, number>> = {};
  for (const turn of turns) {
    counts[turn.policy_mode] ??= {};
    const value = String(turn[key]);
    counts[turn.policy_mode][value] = (counts[turn.policy_mode][value] ?? 0) + 1;
  }
  return counts;
}

export function userTurns(turns: TurnCompletedPayload[], userId: string | null): TurnCompletedPayload[] {
  if (!userId) return turns;
  return turns.filter((turn) => turn.user_id === userId);
}

export function uniqueValues<T extends keyof TurnCompletedPayload>(turns: TurnCompletedPayload[], key: T): string[] {
  return Array.from(new Set(turns.map((turn) => String(turn[key])))).sort();
}

export function formatNumber(value: number | undefined, digits = 3): string {
  if (value === undefined || Number.isNaN(value)) return '0.000';
  return value.toFixed(digits);
}

export function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return sum(values) / values.length;
}

export function sum(values: number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

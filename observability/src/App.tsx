import { useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import './styles.css';
import {
  detectDatasetKind,
  filterKnowledgeSteps,
  filterLegacyTurns,
  formatNumber,
  knowledgePolicyCounts,
  knowledgePolicyOrder,
  knowledgeStepsFromArtifact,
  knowledgeStepsFromEvents,
  legacyPolicyCounts,
  legacyPolicyOrder,
  legacyTurnsFromArtifact,
  legacyTurnsFromEvents,
  parseJsonl,
  summarizeKnowledgeSteps,
  summarizeLegacyTurns,
  uniqueKnowledgeUsers,
  uniqueLegacyUsers,
} from './data';
import type {
  ExperimentArtifact,
  ExperimentEvent,
  KnowledgeExperimentArtifact,
  KnowledgePolicySummary,
  KnowledgeStepPayload,
  LegacyPolicySummary,
  LegacyTurnCompletedPayload,
} from './types';

const streamPath = '/knowledge_policy_comparison_stream.jsonl';

export default function App() {
  const [events, setEvents] = useState<ExperimentEvent[]>([]);
  const [artifact, setArtifact] = useState<ExperimentArtifact | null>(null);
  const [isPolling, setIsPolling] = useState(true);
  const [selectedEntryId, setSelectedEntryId] = useState<string | null>(null);
  const [selectedUser, setSelectedUser] = useState<string | null>(null);
  const [lastLoaded, setLastLoaded] = useState<string>('waiting for stream');

  useEffect(() => {
    if (!isPolling) return;
    const load = async () => {
      try {
        const response = await fetch(`${streamPath}?t=${Date.now()}`, { cache: 'no-store' });
        if (!response.ok) return;
        const text = await response.text();
        const parsed = parseJsonl(text);
        setEvents(parsed);
        setLastLoaded(new Date().toLocaleTimeString());
      } catch {
        setLastLoaded('stream unavailable');
      }
    };
    load();
    const interval = window.setInterval(load, 1500);
    return () => window.clearInterval(interval);
  }, [isPolling]);

  const datasetKind = useMemo(() => detectDatasetKind(events, artifact), [events, artifact]);
  const legacyTurns = useMemo(() => {
    if (datasetKind !== 'legacy') return [];
    if (artifact && 'results' in artifact && !('scenario' in artifact)) {
      return legacyTurnsFromArtifact(artifact);
    }
    return legacyTurnsFromEvents(events);
  }, [artifact, datasetKind, events]);
  const knowledgeSteps = useMemo(() => {
    if (datasetKind !== 'knowledge') return [];
    if (artifact && 'scenario' in artifact) {
      return knowledgeStepsFromArtifact(artifact as KnowledgeExperimentArtifact);
    }
    return knowledgeStepsFromEvents(events);
  }, [artifact, datasetKind, events]);

  const filteredLegacyTurns = useMemo(() => filterLegacyTurns(legacyTurns, selectedUser), [legacyTurns, selectedUser]);
  const filteredKnowledgeSteps = useMemo(() => filterKnowledgeSteps(knowledgeSteps, selectedUser), [knowledgeSteps, selectedUser]);
  const legacySummary = useMemo(() => summarizeLegacyTurns(filteredLegacyTurns), [filteredLegacyTurns]);
  const knowledgeSummary = useMemo(() => summarizeKnowledgeSteps(filteredKnowledgeSteps), [filteredKnowledgeSteps]);
  const users = useMemo(
    () => (datasetKind === 'knowledge' ? uniqueKnowledgeUsers(knowledgeSteps) : uniqueLegacyUsers(legacyTurns)),
    [datasetKind, knowledgeSteps, legacyTurns],
  );

  const selectedLegacyTurn = useMemo(() => {
    if (datasetKind !== 'legacy') return null;
    if (!selectedEntryId) return filteredLegacyTurns.at(-1) ?? null;
    return filteredLegacyTurns.find((turn) => legacyTurnId(turn) === selectedEntryId) ?? filteredLegacyTurns.at(-1) ?? null;
  }, [datasetKind, filteredLegacyTurns, selectedEntryId]);
  const selectedKnowledgeStep = useMemo(() => {
    if (datasetKind !== 'knowledge') return null;
    if (!selectedEntryId) return filteredKnowledgeSteps.at(-1) ?? null;
    return filteredKnowledgeSteps.find((step) => knowledgeStepId(step) === selectedEntryId) ?? filteredKnowledgeSteps.at(-1) ?? null;
  }, [datasetKind, filteredKnowledgeSteps, selectedEntryId]);

  async function handleArtifactUpload(file: File | null) {
    if (!file) return;
    const text = await file.text();
    const parsed = JSON.parse(text) as ExperimentArtifact;
    setArtifact(parsed);
    setIsPolling(false);
    setLastLoaded(`loaded ${file.name}`);
  }

  const totalRows = datasetKind === 'knowledge' ? filteredKnowledgeSteps.length : filteredLegacyTurns.length;
  const policyCount = datasetKind === 'knowledge' ? Object.keys(knowledgeSummary).length : Object.keys(legacySummary).length;

  return (
    <main className="shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Emotiv Learn Observatory</p>
          <h1>{datasetKind === 'knowledge' ? 'Knowledge policy console' : 'Policy experiment console'}</h1>
          <p className="subtle">
            {datasetKind === 'knowledge'
              ? 'Watch knowledge growth, goal coverage, EEG-matched personalization, and step-by-step learner progression.'
              : 'A quiet control surface for watching Tutor LLM calls, hidden-student transitions, observable rewards, and policy updates.'}
          </p>
        </div>
        <div className="controls">
          <button className={isPolling ? 'active' : ''} onClick={() => setIsPolling((value) => !value)}>
            {isPolling ? 'Polling stream' : 'Stream paused'}
          </button>
          <label className="fileButton">
            Load artifact
            <input type="file" accept=".json" onChange={(event) => handleArtifactUpload(event.target.files?.[0] ?? null)} />
          </label>
        </div>
      </header>

      <section className="statusGrid">
        <Metric label={datasetKind === 'knowledge' ? 'step events' : 'turn events'} value={String(totalRows)} />
        <Metric label="users" value={String(users.length)} />
        <Metric label="policies" value={String(policyCount)} />
        <Metric label="last update" value={lastLoaded} mono />
      </section>

      <section className="toolbar">
        <span>Focus</span>
        <button className={selectedUser === null ? 'active' : ''} onClick={() => setSelectedUser(null)}>
          all users
        </button>
        {users.map((user) => (
          <button key={user} className={selectedUser === user ? 'active' : ''} onClick={() => setSelectedUser(user)}>
            {user}
          </button>
        ))}
      </section>

      {datasetKind === 'knowledge' ? (
        <KnowledgeLayout
          steps={filteredKnowledgeSteps}
          summary={knowledgeSummary}
          selectedStep={selectedKnowledgeStep}
          onSelect={setSelectedEntryId}
        />
      ) : (
        <LegacyLayout
          turns={filteredLegacyTurns}
          summary={legacySummary}
          selectedTurn={selectedLegacyTurn}
          onSelect={setSelectedEntryId}
        />
      )}
    </main>
  );
}

function KnowledgeLayout({
  steps,
  summary,
  selectedStep,
  onSelect,
}: {
  steps: KnowledgeStepPayload[];
  summary: Record<string, KnowledgePolicySummary>;
  selectedStep: KnowledgeStepPayload | null;
  onSelect: (id: string) => void;
}) {
  const actionsByPolicy = useMemo(() => knowledgePolicyCounts(steps, 'action_id'), [steps]);
  const conceptsByPolicy = useMemo(() => knowledgePolicyCounts(steps, 'concept_id'), [steps]);
  const responsesByPolicy = useMemo(() => knowledgePolicyCounts(steps, 'response_type'), [steps]);

  return (
    <section className="layout">
      <div className="leftColumn">
        <Panel title="Policy Comparison" quiet>
          <KnowledgePolicyTable summary={summary} steps={steps} />
        </Panel>
        <Panel title="Action Mix">
          <Distribution counts={actionsByPolicy} order={knowledgePolicyOrder} />
        </Panel>
        <Panel title="Concept Mix">
          <Distribution counts={conceptsByPolicy} order={knowledgePolicyOrder} />
        </Panel>
        <Panel title="Response Mix">
          <Distribution counts={responsesByPolicy} order={knowledgePolicyOrder} />
        </Panel>
      </div>

      <Panel title="Goal / Knowledge Evolution" className="evolutionPanel">
        <KnowledgeEvolutionCharts
          steps={steps}
          selectedId={selectedStep ? knowledgeStepId(selectedStep) : null}
          onSelect={onSelect}
        />
      </Panel>

      <div className="rightColumn">
        <Panel title="Step Log">
          {selectedStep ? <KnowledgeStepDetail step={selectedStep} /> : <EmptyState />}
        </Panel>
      </div>
    </section>
  );
}

function LegacyLayout({
  turns,
  summary,
  selectedTurn,
  onSelect,
}: {
  turns: LegacyTurnCompletedPayload[];
  summary: Record<string, LegacyPolicySummary>;
  selectedTurn: LegacyTurnCompletedPayload | null;
  onSelect: (id: string) => void;
}) {
  const actionsByPolicy = useMemo(() => legacyPolicyCounts(turns, 'action_id'), [turns]);
  const responsesByPolicy = useMemo(() => legacyPolicyCounts(turns, 'response_type'), [turns]);

  return (
    <section className="layout">
      <div className="leftColumn">
        <Panel title="Policy Comparison" quiet>
          <LegacyPolicyTable summary={summary} turns={turns} />
        </Panel>
        <Panel title="Action Mix">
          <Distribution counts={actionsByPolicy} order={legacyPolicyOrder} />
        </Panel>
        <Panel title="Response Mix">
          <Distribution counts={responsesByPolicy} order={legacyPolicyOrder} />
        </Panel>
      </div>

      <Panel title="Reward / Oracle Evolution" className="evolutionPanel">
        <LegacyEvolutionCharts turns={turns} selectedId={selectedTurn ? legacyTurnId(selectedTurn) : null} onSelect={onSelect} />
      </Panel>

      <div className="rightColumn">
        <Panel title="Turn Log">
          {selectedTurn ? <LegacyTurnDetail turn={selectedTurn} /> : <EmptyState />}
        </Panel>
      </div>
    </section>
  );
}

function Metric({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong className={mono ? 'mono' : ''}>{value}</strong>
    </div>
  );
}

function Panel({ title, children, className = '', quiet = false }: { title: string; children: ReactNode; className?: string; quiet?: boolean }) {
  return (
    <section className={`panel ${quiet ? 'quiet' : ''} ${className}`}>
      <div className="panelHeader">
        <h2>{title}</h2>
      </div>
      {children}
    </section>
  );
}

function KnowledgePolicyTable({ summary, steps }: { summary: Record<string, KnowledgePolicySummary>; steps: KnowledgeStepPayload[] }) {
  return (
    <div className="table">
      <div className="tableRow head knowledgeHead">
        <span>Policy</span>
        <span>Coverage</span>
        <span>Quality</span>
        <span>Goal</span>
      </div>
      {knowledgePolicyOrder
        .filter((policy) => summary[policy])
        .map((policy) => {
          const rows = steps.filter((step) => step.policy_name === policy);
          return (
            <div className="tableRow knowledgeRow" key={policy}>
              <span className="policyName">{policy}</span>
              <span>{formatNumber(summary[policy].avg_final_goal_coverage)}</span>
              <span>{formatNumber(summary[policy].avg_final_knowledge_quality)}</span>
              <span>{summary[policy].avg_steps_to_goal === null ? '—' : formatNumber(summary[policy].avg_steps_to_goal)}</span>
              <div className="bar" style={{ '--value': String(Math.min(summary[policy].avg_final_goal_coverage, 1)) } as React.CSSProperties} />
              <small>{rows.length} steps</small>
            </div>
          );
        })}
    </div>
  );
}

function LegacyPolicyTable({ summary, turns }: { summary: Record<string, LegacyPolicySummary>; turns: LegacyTurnCompletedPayload[] }) {
  return (
    <div className="table">
      <div className="tableRow head">
        <span>Policy</span>
        <span>Reward</span>
        <span>Oracle</span>
        <span>Checkpoint</span>
      </div>
      {legacyPolicyOrder
        .filter((policy) => summary[policy])
        .map((policy) => {
          const rows = turns.filter((turn) => turn.policy_mode === policy);
          return (
            <div className="tableRow" key={policy}>
              <span className="policyName">{policy}</span>
              <span>{formatNumber(summary[policy].average_reward)}</span>
              <span>{formatNumber(summary[policy].total_oracle_mastery_gain)}</span>
              <span>{formatNumber(summary[policy].checkpoint_accuracy)}</span>
              <div className="bar" style={{ '--value': String(Math.min(summary[policy].average_reward / 1.5, 1)) } as React.CSSProperties} />
              <small>{rows.length} turns</small>
            </div>
          );
        })}
    </div>
  );
}

function Distribution({ counts, order }: { counts: Record<string, Record<string, number>>; order: string[] }) {
  return (
    <div className="distribution">
      {order
        .filter((policy) => counts[policy])
        .map((policy) => {
          const entries = Object.entries(counts[policy]).sort((a, b) => b[1] - a[1]).slice(0, 4);
          const total = entries.reduce((sum, [, value]) => sum + value, 0) || 1;
          return (
            <div className="distRow" key={policy}>
              <span className="policyName">{policy}</span>
              <div className="pills">
                {entries.map(([label, value]) => (
                  <span key={label} className="pill">
                    {label} <b>{Math.round((value / total) * 100)}%</b>
                  </span>
                ))}
              </div>
            </div>
          );
        })}
    </div>
  );
}

function KnowledgeStepDetail({ step }: { step: KnowledgeStepPayload }) {
  const eeg = step.policy_metadata?.eeg;
  const matchedWindow = eeg?.metadata?.window_id ? String(eeg.metadata.window_id) : '—';
  const subjectId = eeg?.metadata?.subject_id ? String(eeg.metadata.subject_id) : '—';
  const achievedClaims = step.evaluation.achieved_goal_claims ?? [];
  const missingClaims = step.evaluation.missing_goal_claims ?? [];
  return (
    <div className="detail">
      <div className="detailGrid">
        <Metric label="policy" value={step.policy_name} mono />
        <Metric label="action" value={step.action_id} mono />
        <Metric label="coverage" value={formatNumber(step.goal_coverage)} />
        <Metric label="quality" value={formatNumber(step.knowledge_quality)} />
      </div>
      <div className="detailGrid">
        <Metric label="response" value={step.response_type} mono />
        <Metric label="confidence" value={formatNumber(step.knowledge_turn.self_reported_confidence)} />
        <Metric label="subject" value={subjectId} mono />
        <Metric label="window" value={matchedWindow} mono />
      </div>
      <article>
        <h3>Tutor</h3>
        <p>{step.tutor_message}</p>
      </article>
      <article>
        <h3>Learner reprompt</h3>
        <p>{step.knowledge_turn.reprompt ?? 'No reprompt'}</p>
      </article>
      <article>
        <h3>Knowledge base snapshot</h3>
        <ul className="claimList">
          {(step.knowledge_base ?? []).map((claim) => (
            <li key={claim}>{claim}</li>
          ))}
        </ul>
      </article>
      <article>
        <h3>Goal progress</h3>
        <p>Achieved: {achievedClaims.length ? achievedClaims.join(' | ') : 'None yet'}</p>
        <p>Missing: {missingClaims.length ? missingClaims.join(' | ') : 'Goal reached'}</p>
      </article>
      <article>
        <h3>State / EEG</h3>
        <pre>
          {JSON.stringify(
            {
              concept_mastery: step.knowledge_turn.state_after.concept_mastery,
              interpreted: step.policy_metadata?.interpreted ?? null,
              eeg: eeg
                ? {
                    feature_names: eeg.feature_names,
                    features: eeg.features,
                    metadata: eeg.metadata,
                  }
                : null,
            },
            null,
            2,
          )}
        </pre>
      </article>
    </div>
  );
}

function LegacyTurnDetail({ turn }: { turn: LegacyTurnCompletedPayload }) {
  const transition = turn.student_transition;
  return (
    <div className="detail">
      <div className="detailGrid">
        <Metric label="policy" value={turn.policy_mode} mono />
        <Metric label="action" value={turn.action_id} mono />
        <Metric label="reward" value={formatNumber(turn.reward)} />
        <Metric label="oracle gain" value={formatNumber(turn.oracle_mastery_gain)} />
      </div>
      <article>
        <h3>Tutor</h3>
        <p>{turn.tutor_message}</p>
      </article>
      <article>
        <h3>Student</h3>
        <p>{String(turn.student_response.student_message ?? 'No message')}</p>
      </article>
      <article>
        <h3>Transition</h3>
        <pre>{JSON.stringify({
          evaluation: transition.evaluation,
          response_type_probs: transition.response_type_probs,
          observable_signals: transition.observable_signals,
          update_trace: turn.update_trace,
        }, null, 2)}</pre>
      </article>
    </div>
  );
}

type LegacyPoint = { turnIndex: number; value: number; turn: LegacyTurnCompletedPayload };
type KnowledgePoint = { stepIndex: number; value: number; step: KnowledgeStepPayload };

const policyColors: Record<string, string> = {
  personalized: '#48c17a',
  generic: '#66a7c8',
  fixed_no_change: '#d6a657',
  random: '#b58cff',
};

function KnowledgeEvolutionCharts({
  steps,
  selectedId,
  onSelect,
}: {
  steps: KnowledgeStepPayload[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  if (steps.length === 0) return <EmptyState />;

  const policies = knowledgePolicyOrder.filter((policy) => steps.some((step) => step.policy_name === policy));
  return (
    <div className="evolutionChart">
      <div className="legend">
        {policies.map((policy) => (
          <span key={policy}><i style={{ background: policyColors[policy] }} /> {policy}</span>
        ))}
      </div>
      <KnowledgeMetricChart title="Goal Coverage" metric="goal_coverage" steps={steps} selectedId={selectedId} onSelect={onSelect} />
      <KnowledgeMetricChart title="Knowledge Quality" metric="knowledge_quality" steps={steps} selectedId={selectedId} onSelect={onSelect} />
    </div>
  );
}

function KnowledgeMetricChart({
  title,
  metric,
  steps,
  selectedId,
  onSelect,
}: {
  title: string;
  metric: 'goal_coverage' | 'knowledge_quality';
  steps: KnowledgeStepPayload[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const width = 980;
  const height = 292;
  const left = 54;
  const right = 26;
  const top = 28;
  const bottom = 38;
  const valuesByPolicy = knowledgePointsByPolicy(steps, metric);
  const allPoints = Object.values(valuesByPolicy).flat();
  const minValue = 0;
  const maxValue = Math.max(...allPoints.map((point) => point.value), 0.01);
  const stepIndexes = Array.from(new Set(allPoints.map((point) => point.stepIndex))).sort((a, b) => a - b);
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const xForStep = (stepIndex: number) => {
    const position = stepIndexes.indexOf(stepIndex);
    return left + (stepIndexes.length <= 1 ? 0 : (position / (stepIndexes.length - 1)) * plotWidth);
  };
  const yForValue = (value: number) => top + (1 - (value - minValue) / Math.max(maxValue - minValue, 0.001)) * plotHeight;

  return (
    <section className="metricChart">
      <div className="chartTitle">
        <h3>{title}</h3>
        <span className="mono">{formatNumber(minValue)} - {formatNumber(maxValue)}</span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${title} by policy over steps`}>
        <line className="chartGuide" x1={left} x2={width - right} y1={yForValue(0)} y2={yForValue(0)} />
        <line className="chartDivider" x1={left} x2={width - right} y1={height - bottom} y2={height - bottom} />
        {stepIndexes.map((stepIndex) => (
          <text className="turnTick" key={stepIndex} x={xForStep(stepIndex)} y={height - 12}>
            {stepIndex}
          </text>
        ))}
        {knowledgePolicyOrder.map((policy) => {
          const points = valuesByPolicy[policy] ?? [];
          if (points.length === 0) return null;
          const coordinates = points.map((point) => ({ x: xForStep(point.stepIndex), y: yForValue(point.value) }));
          return (
            <g key={policy}>
              <path className="policyLine" d={smoothPath(coordinates)} style={{ stroke: policyColors[policy] }} />
              {points.map((point) => {
                const id = knowledgeStepId(point.step);
                const selected = selectedId === id;
                const x = xForStep(point.stepIndex);
                const y = yForValue(point.value);
                return (
                  <g key={`${metric}:${id}`} className={`chartPointGroup ${selected ? 'selected' : ''}`} onClick={() => onSelect(id)}>
                    <title>{`${policy} s${point.stepIndex} ${title.toLowerCase()}: ${formatNumber(point.value)}`}</title>
                    <circle className="pointTarget" cx={x} cy={y} r="12" />
                    <circle className="policyPoint" cx={x} cy={y} r={selected ? 6 : 4} style={{ fill: policyColors[policy] }} />
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
    </section>
  );
}

function LegacyEvolutionCharts({
  turns,
  selectedId,
  onSelect,
}: {
  turns: LegacyTurnCompletedPayload[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  if (turns.length === 0) return <EmptyState />;

  const policies = legacyPolicyOrder.filter((policy) => turns.some((turn) => turn.policy_mode === policy));
  return (
    <div className="evolutionChart">
      <div className="legend">
        {policies.map((policy) => (
          <span key={policy}><i style={{ background: policyColors[policy] }} /> {policy}</span>
        ))}
      </div>
      <LegacyMetricChart title="Reward" metric="reward" turns={turns} selectedId={selectedId} onSelect={onSelect} />
      <LegacyMetricChart title="Oracle Gain" metric="oracle" turns={turns} selectedId={selectedId} onSelect={onSelect} />
    </div>
  );
}

function LegacyMetricChart({
  title,
  metric,
  turns,
  selectedId,
  onSelect,
}: {
  title: string;
  metric: 'reward' | 'oracle';
  turns: LegacyTurnCompletedPayload[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const width = 980;
  const height = 292;
  const left = 54;
  const right = 26;
  const top = 28;
  const bottom = 38;
  const valuesByPolicy = legacyPointsByPolicy(turns, metric);
  const allPoints = Object.values(valuesByPolicy).flat();
  const minValue = metric === 'reward' ? Math.min(0, ...allPoints.map((point) => point.value)) : 0;
  const maxValue = Math.max(...allPoints.map((point) => point.value), metric === 'reward' ? 1 : 0.01);
  const turnIndexes = Array.from(new Set(allPoints.map((point) => point.turnIndex))).sort((a, b) => a - b);
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const xForTurn = (turnIndex: number) => {
    const position = turnIndexes.indexOf(turnIndex);
    return left + (turnIndexes.length <= 1 ? 0 : (position / (turnIndexes.length - 1)) * plotWidth);
  };
  const yForValue = (value: number) => top + (1 - (value - minValue) / Math.max(maxValue - minValue, 0.001)) * plotHeight;

  return (
    <section className="metricChart">
      <div className="chartTitle">
        <h3>{title}</h3>
        <span className="mono">{formatNumber(minValue)} - {formatNumber(maxValue)}</span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${title} by policy over turns`}>
        <line className="chartGuide" x1={left} x2={width - right} y1={yForValue(0)} y2={yForValue(0)} />
        <line className="chartDivider" x1={left} x2={width - right} y1={height - bottom} y2={height - bottom} />
        {turnIndexes.map((turnIndex) => (
          <text className="turnTick" key={turnIndex} x={xForTurn(turnIndex)} y={height - 12}>
            {turnIndex}
          </text>
        ))}
        {legacyPolicyOrder.map((policy) => {
          const points = valuesByPolicy[policy] ?? [];
          if (points.length === 0) return null;
          const coordinates = points.map((point) => ({ x: xForTurn(point.turnIndex), y: yForValue(point.value) }));
          return (
            <g key={policy}>
              <path className="policyLine" d={smoothPath(coordinates)} style={{ stroke: policyColors[policy] }} />
              {points.map((point) => {
                const id = legacyTurnId(point.turn);
                const selected = selectedId === id;
                const x = xForTurn(point.turnIndex);
                const y = yForValue(point.value);
                return (
                  <g key={`${metric}:${id}`} className={`chartPointGroup ${selected ? 'selected' : ''}`} onClick={() => onSelect(id)}>
                    <title>{`${policy} t${point.turnIndex} ${title.toLowerCase()}: ${formatNumber(point.value)}`}</title>
                    <circle className="pointTarget" cx={x} cy={y} r="12" />
                    <circle className="policyPoint" cx={x} cy={y} r={selected ? 6 : 4} style={{ fill: policyColors[policy] }} />
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
    </section>
  );
}

function legacyPointsByPolicy(turns: LegacyTurnCompletedPayload[], metric: 'reward' | 'oracle'): Record<string, LegacyPoint[]> {
  const grouped: Record<string, Map<number, LegacyTurnCompletedPayload[]>> = {};
  for (const turn of turns) {
    grouped[turn.policy_mode] ??= new Map();
    const rows = grouped[turn.policy_mode].get(turn.turn_index) ?? [];
    rows.push(turn);
    grouped[turn.policy_mode].set(turn.turn_index, rows);
  }

  const result: Record<string, LegacyPoint[]> = {};
  for (const [policy, turnsByIndex] of Object.entries(grouped)) {
    result[policy] = Array.from(turnsByIndex.entries())
      .sort(([a], [b]) => a - b)
      .map(([turnIndex, rows]) => ({
        turnIndex,
        value: rows.reduce((sum, turn) => sum + (metric === 'reward' ? turn.reward : turn.oracle_mastery_gain), 0) / rows.length,
        turn: rows[0],
      }));
  }
  return result;
}

function knowledgePointsByPolicy(
  steps: KnowledgeStepPayload[],
  metric: 'goal_coverage' | 'knowledge_quality',
): Record<string, KnowledgePoint[]> {
  const grouped: Record<string, Map<number, KnowledgeStepPayload[]>> = {};
  for (const step of steps) {
    grouped[step.policy_name] ??= new Map();
    const rows = grouped[step.policy_name].get(step.step_index) ?? [];
    rows.push(step);
    grouped[step.policy_name].set(step.step_index, rows);
  }

  const result: Record<string, KnowledgePoint[]> = {};
  for (const [policy, stepsByIndex] of Object.entries(grouped)) {
    result[policy] = Array.from(stepsByIndex.entries())
      .sort(([a], [b]) => a - b)
      .map(([stepIndex, rows]) => ({
        stepIndex,
        value: rows.reduce((sum, step) => sum + (metric === 'goal_coverage' ? step.goal_coverage : step.knowledge_quality), 0) / rows.length,
        step: rows[0],
      }));
  }
  return result;
}

function smoothPath(points: { x: number; y: number }[]): string {
  if (points.length === 0) return '';
  if (points.length === 1) return `M ${points[0].x} ${points[0].y}`;
  const commands = [`M ${points[0].x} ${points[0].y}`];
  for (let index = 0; index < points.length - 1; index += 1) {
    const current = points[index];
    const next = points[index + 1];
    const controlX = (current.x + next.x) / 2;
    commands.push(`C ${controlX} ${current.y}, ${controlX} ${next.y}, ${next.x} ${next.y}`);
  }
  return commands.join(' ');
}

function EmptyState() {
  return <div className="empty">No experiment events yet. Start a run or load an artifact.</div>;
}

function legacyTurnId(turn: LegacyTurnCompletedPayload): string {
  return `${turn.policy_mode}:${turn.user_id}:${turn.turn_index}`;
}

function knowledgeStepId(step: KnowledgeStepPayload): string {
  return `${step.policy_name}:${step.user_id}:${step.step_index}`;
}

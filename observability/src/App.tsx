import { useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import './styles.css';
import {
  formatNumber,
  parseJsonl,
  policyOrder,
  policyCounts,
  summarizeTurns,
  turnsFromEvents,
  uniqueValues,
  userTurns,
} from './data';
import type { ExperimentEvent, TurnCompletedPayload } from './types';

export default function App() {
  const [events, setEvents] = useState<ExperimentEvent[]>([]);
  const [selectedTurnId, setSelectedTurnId] = useState<string | null>(null);
  const [selectedUser, setSelectedUser] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>('No file loaded');
  const [dragActive, setDragActive] = useState(false);

  const eventTurns = useMemo(() => turnsFromEvents(events), [events]);
  const turns = eventTurns;
  const userFilteredTurns = useMemo(() => userTurns(turns, selectedUser), [turns, selectedUser]);
  const summary = useMemo(() => summarizeTurns(userFilteredTurns), [userFilteredTurns]);
  const users = useMemo(() => uniqueValues(turns, 'user_id'), [turns]);
  const actionsByPolicy = useMemo(() => policyCounts(userFilteredTurns, 'action_id'), [userFilteredTurns]);
  const contentMixByPolicy = useMemo(() => contentMixCounts(userFilteredTurns), [userFilteredTurns]);
  const selectedTurn = useMemo(() => {
    if (!selectedTurnId) return userFilteredTurns.at(-1) ?? null;
    return userFilteredTurns.find((turn) => turnId(turn) === selectedTurnId) ?? userFilteredTurns.at(-1) ?? null;
  }, [userFilteredTurns, selectedTurnId]);
  const experimentMeta = useMemo(() => {
    const start = events.find((event) => event.event_type === 'experiment_started');
    return start?.payload as Record<string, unknown> | undefined;
  }, [events]);
  const metrics = useMemo(() => summarizeMetrics(userFilteredTurns), [userFilteredTurns]);

  async function handleJsonlUpload(file: File | null) {
    if (!file) return;
    const text = await file.text();
    const parsed = parseJsonl(text);
    setEvents(parsed);
    setFileName(file.name);
    setSelectedTurnId(null);
  }

  function handleDrop(event: React.DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragActive(false);
    void handleJsonlUpload(event.dataTransfer.files?.[0] ?? null);
  }

  return (
    <main className="shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Interpreted Policy Report</p>
          <h1>Live policy events report</h1>
          <p className="subtle">
            Drag in an interpreted JSONL log and inspect policy behavior, reward/oracle movement, interpreted signals, and EEG proxy changes.
          </p>
        </div>
        <div className="controls">
          <label className="fileButton">
            Load JSONL
            <input type="file" accept=".jsonl" onChange={(event) => void handleJsonlUpload(event.target.files?.[0] ?? null)} />
          </label>
          <span className="fileLabel">{fileName}</span>
        </div>
      </header>

      {turns.length === 0 ? (
        <section
          className={`dropzone ${dragActive ? 'dragActive' : ''}`}
          onDragEnter={(event) => {
            event.preventDefault();
            setDragActive(true);
          }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={(event) => {
            event.preventDefault();
            setDragActive(false);
          }}
          onDrop={handleDrop}
        >
          <p className="dropEyebrow">Drop interpreted events</p>
          <h2>Load a `.jsonl` report</h2>
          <p>Best with `turn_completed` payloads that include interpreted signals, EEG windows, proxy estimates, and student transitions.</p>
        </section>
      ) : (
        <>
          <section className="reportTabs">
            <span className="active">Report</span>
            <span>Policies</span>
            <span>Turns</span>
            <span>Signals</span>
          </section>

      <section className="statusGrid">
        <Metric label="turns" value={String(experimentMeta?.turns ?? userFilteredTurns.length)} />
        <Metric label="users" value={String(users.length)} />
        <Metric label="avg reward" value={formatNumber(metrics.averageReward)} />
        <Metric label="avg oracle gain" value={formatNumber(metrics.averageOracle)} />
        <Metric label="checkpoint rate" value={`${Math.round(metrics.checkpointRate * 100)}%`} />
        <Metric label="avg workload" value={formatNumber(metrics.averageWorkload)} />
      </section>

      <section className="toolbar">
        <span>Report Focus</span>
        <button className={selectedUser === null ? 'active' : ''} onClick={() => setSelectedUser(null)}>
          all users
        </button>
        {users.map((user) => (
          <button key={user} className={selectedUser === user ? 'active' : ''} onClick={() => setSelectedUser(user)}>
            {user}
          </button>
        ))}
      </section>

      <section className="layout">
        <div className="leftColumn">
          <Panel title="Policy Snapshot" quiet>
            <PolicyTable summary={summary} turns={userFilteredTurns} />
          </Panel>
          <Panel title="Action Mix">
            <Distribution counts={actionsByPolicy} />
          </Panel>
          <Panel title="Content Mix">
            <Distribution counts={contentMixByPolicy} />
          </Panel>
          <Panel title="Completed Turns">
            <TurnList turns={userFilteredTurns} selectedId={selectedTurn ? turnId(selectedTurn) : null} onSelect={setSelectedTurnId} />
          </Panel>
        </div>

        <Panel title="Reward / Oracle Evolution" className="evolutionPanel">
          <PolicyEvolutionCharts turns={userFilteredTurns} selectedId={selectedTurn ? turnId(selectedTurn) : null} onSelect={setSelectedTurnId} />
        </Panel>

        <div className="rightColumn">
          <Panel title="Turn Log">
            {selectedTurn ? <TurnDetail turn={selectedTurn} /> : <EmptyState />}
          </Panel>
        </div>
      </section>
        </>
      )}
    </main>
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

function PolicyTable({ summary, turns }: { summary: ReturnType<typeof summarizeTurns>; turns: TurnCompletedPayload[] }) {
  return (
    <div className="table">
      <div className="tableRow head">
        <span>Policy</span>
        <span>Reward</span>
        <span>Oracle</span>
        <span>Checkpoint</span>
      </div>
      {policyOrder
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

function Distribution({ counts }: { counts: Record<string, Record<string, number>> }) {
  const policies = policyOrder.filter((policy) => counts[policy] && Object.keys(counts[policy]).length > 0);
  if (policies.length === 0) return <EmptyState />;

  return (
    <div className="table">
      {policies.map((policy) => {
        const entries = Object.entries(counts[policy]).sort((a, b) => b[1] - a[1]);
        const total = entries.reduce((sum, [, value]) => sum + value, 0);
        return (
          <div key={policy} className="distributionBlock">
            <div className="tableRow head">
              <span>{policy}</span>
              <span>{total} turns</span>
            </div>
            {entries.map(([label, value]) => (
              <div className="tableRow" key={`${policy}:${label}`}>
                <span className="mono">{label}</span>
                <span>{value}</span>
                <div
                  className="bar"
                  style={{ '--value': String(total === 0 ? 0 : value / total) } as React.CSSProperties}
                />
                <small>{Math.round((total === 0 ? 0 : value / total) * 100)}%</small>
              </div>
            ))}
          </div>
        );
      })}
    </div>
  );
}

function TurnList({
  turns,
  selectedId,
  onSelect,
}: {
  turns: TurnCompletedPayload[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="turnList">
      {[...turns]
        .sort((a, b) => a.turn_index - b.turn_index || a.user_id.localeCompare(b.user_id) || a.policy_mode.localeCompare(b.policy_mode))
        .map((turn) => {
          const id = turnId(turn);
          const workload = getWorkload(turn);
          return (
            <button key={id} className={`turnItem ${selectedId === id ? 'selected' : ''}`} onClick={() => onSelect(id)}>
              <span className="mono">t{turn.turn_index}</span>
              <span>{turn.user_id}</span>
              <span>{turn.policy_mode}</span>
              <span className={turn.reward >= 0 ? 'positive' : 'negative'}>{formatNumber(turn.reward)}</span>
              <span>{formatNumber(turn.oracle_mastery_gain)}</span>
              <span>{formatNumber(workload)}</span>
            </button>
          );
        })}
    </div>
  );
}

function TurnDetail({ turn }: { turn: TurnCompletedPayload }) {
  const transition = turn.student_transition;
  return (
    <div className="detail">
      <div className="detailGrid">
        <Metric label="policy" value={turn.policy_mode} mono />
        <Metric label="action" value={turn.action_id} mono />
        <Metric label="reward" value={formatNumber(turn.reward)} />
        <Metric label="oracle gain" value={formatNumber(turn.oracle_mastery_gain)} />
        <Metric label="workload" value={formatNumber(getWorkload(turn))} />
        <Metric label="engagement" value={formatNumber(getEngagement(turn))} />
      </div>
      <div className="detailGrid">
        <Metric label="dominant mix" value={dominantContentMix(turn)} mono />
        <Metric label="description" value={formatNumber(getContentMix(turn).text_description)} />
        <Metric label="examples" value={formatNumber(getContentMix(turn).text_examples)} />
        <Metric label="visual" value={formatNumber(getContentMix(turn).visual)} />
      </div>
      <article>
        <h3>Learner Style</h3>
        <pre>{JSON.stringify(getLearningStyle(turn), null, 2)}</pre>
      </article>
      <article>
        <h3>Hidden Knowledge State</h3>
        <pre>{JSON.stringify(getHiddenKnowledgeState(turn), null, 2)}</pre>
      </article>
      <article>
        <h3>Claims Understood</h3>
        <pre>{JSON.stringify(getClaimsUnderstood(turn), null, 2)}</pre>
      </article>
      <article>
        <h3>Hidden Neuro State</h3>
        <pre>{JSON.stringify(getHiddenNeuroState(turn), null, 2)}</pre>
      </article>
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
          content_mix: getContentMix(turn),
          response_type_probs: transition.response_type_probs,
          observable_signals: transition.observable_signals,
          update_trace: turn.update_trace,
        }, null, 2)}</pre>
      </article>
    </div>
  );
}

type PolicyPoint = {
  turnIndex: number;
  value: number;
  turn: TurnCompletedPayload;
};

const policyColors: Record<string, string> = {
  personalized: '#48c17a',
  generic: '#66a7c8',
  fixed_no_change: '#d6a657',
  random: '#b58cff',
};

function PolicyEvolutionCharts({
  turns,
  selectedId,
  onSelect,
}: {
  turns: TurnCompletedPayload[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  if (turns.length === 0) return <EmptyState />;

  const policies = policyOrder.filter((policy) => turns.some((turn) => turn.policy_mode === policy));

  return (
    <div className="evolutionChart">
      <div className="legend">
        {policies.map((policy) => (
          <span key={policy}><i style={{ background: policyColors[policy] }} /> {policy}</span>
        ))}
      </div>
      <MetricChart title="Reward" metric="reward" turns={turns} selectedId={selectedId} onSelect={onSelect} />
      <MetricChart title="Oracle Gain" metric="oracle" turns={turns} selectedId={selectedId} onSelect={onSelect} />
    </div>
  );
}

function MetricChart({
  title,
  metric,
  turns,
  selectedId,
  onSelect,
}: {
  title: string;
  metric: 'reward' | 'oracle';
  turns: TurnCompletedPayload[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const width = 980;
  const height = 292;
  const left = 54;
  const right = 26;
  const top = 28;
  const bottom = 38;
  const valuesByPolicy = pointsByPolicy(turns, metric);
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
        {policyOrder.map((policy) => {
          const points = valuesByPolicy[policy] ?? [];
          if (points.length === 0) return null;
          const coordinates = points.map((point) => ({ x: xForTurn(point.turnIndex), y: yForValue(point.value) }));
          return (
            <g key={policy}>
              <path className="policyLine" d={smoothPath(coordinates)} style={{ stroke: policyColors[policy] }} />
              {points.map((point) => {
                const id = turnId(point.turn);
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

function pointsByPolicy(turns: TurnCompletedPayload[], metric: 'reward' | 'oracle'): Record<string, PolicyPoint[]> {
  const grouped: Record<string, Map<number, TurnCompletedPayload[]>> = {};
  for (const turn of turns) {
    grouped[turn.policy_mode] ??= new Map();
    const rows = grouped[turn.policy_mode].get(turn.turn_index) ?? [];
    rows.push(turn);
    grouped[turn.policy_mode].set(turn.turn_index, rows);
  }

  const result: Record<string, PolicyPoint[]> = {};
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

function summarizeMetrics(turns: TurnCompletedPayload[]) {
  const checkpoints = turns.filter((turn) => turn.checkpoint_correct !== null);
  return {
    averageReward: average(turns.map((turn) => turn.reward)),
    averageOracle: average(turns.map((turn) => turn.oracle_mastery_gain)),
    checkpointRate: checkpoints.length === 0 ? 0 : checkpoints.filter((turn) => turn.checkpoint_correct === true).length / checkpoints.length,
    averageWorkload: average(turns.map(getWorkload)),
  };
}

function getWorkload(turn: TurnCompletedPayload): number {
  const proxy = (turn as TurnCompletedPayload & { eeg_proxy_estimate?: { workload_estimate?: number } }).eeg_proxy_estimate;
  return proxy?.workload_estimate ?? 0;
}

function getEngagement(turn: TurnCompletedPayload): number {
  const signals = turn.student_transition?.observable_signals as Record<string, unknown> | undefined;
  const value = signals?.engagement_score;
  return typeof value === 'number' ? value : 0;
}

function getContentMix(turn: TurnCompletedPayload): {
  text_description: number;
  text_examples: number;
  visual: number;
} {
  const evaluation = turn.student_transition?.evaluation as Record<string, unknown> | undefined;
  return {
    text_description: typeof evaluation?.content_text_description === 'number' ? evaluation.content_text_description : 0,
    text_examples: typeof evaluation?.content_text_examples === 'number' ? evaluation.content_text_examples : 0,
    visual: typeof evaluation?.content_visual === 'number' ? evaluation.content_visual : 0,
  };
}

function dominantContentMix(turn: TurnCompletedPayload): string {
  const mix = getContentMix(turn);
  const entries = Object.entries(mix).sort((a, b) => b[1] - a[1]);
  return entries[0]?.[0] ?? 'unknown';
}

function getLearningStyle(turn: TurnCompletedPayload): Record<string, number> {
  const hiddenAfter = turn.student_transition?.hidden_state_after as Record<string, unknown> | undefined;
  const knowledgeState = hiddenAfter?.knowledge_state as Record<string, unknown> | undefined;
  const learningStyle = knowledgeState?.learning_style as Record<string, unknown> | undefined;
  return {
    text_description: typeof learningStyle?.text_description === 'number' ? learningStyle.text_description : 0,
    text_examples: typeof learningStyle?.text_examples === 'number' ? learningStyle.text_examples : 0,
    visual: typeof learningStyle?.visual === 'number' ? learningStyle.visual : 0,
  };
}

function getHiddenKnowledgeState(turn: TurnCompletedPayload): Record<string, unknown> {
  const hiddenAfter = turn.student_transition?.hidden_state_after as Record<string, unknown> | undefined;
  const knowledgeState = hiddenAfter?.knowledge_state as Record<string, unknown> | undefined;
  return knowledgeState ?? {};
}

function getClaimsUnderstood(turn: TurnCompletedPayload): Record<string, unknown> {
  const knowledgeState = getHiddenKnowledgeState(turn);
  const claims = knowledgeState.claims_understood as Record<string, unknown> | undefined;
  return claims ?? {};
}

function getHiddenNeuroState(turn: TurnCompletedPayload): Record<string, unknown> {
  const hiddenAfter = turn.student_transition?.hidden_state_after as Record<string, unknown> | undefined;
  const neuroState = hiddenAfter?.neuro_state as Record<string, unknown> | undefined;
  return neuroState ?? {};
}

function contentMixCounts(turns: TurnCompletedPayload[]): Record<string, Record<string, number>> {
  const counts: Record<string, Record<string, number>> = {};
  for (const turn of turns) {
    counts[turn.policy_mode] ??= {};
    const label = dominantContentMix(turn);
    counts[turn.policy_mode][label] = (counts[turn.policy_mode][label] ?? 0) + 1;
  }
  return counts;
}

function average(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function turnId(turn: TurnCompletedPayload): string {
  return `${turn.policy_mode}:${turn.user_id}:${turn.turn_index}`;
}

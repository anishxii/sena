import { useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import './styles.css';
import {
  formatNumber,
  parseJsonl,
  policyCounts,
  policyOrder,
  summarizeTurns,
  turnsFromArtifact,
  turnsFromEvents,
  uniqueValues,
  userTurns,
} from './data';
import type { ExperimentArtifact, ExperimentEvent, TurnCompletedPayload } from './types';

const streamPath = '/live_policy_comparison_stream.jsonl';

export default function App() {
  const [events, setEvents] = useState<ExperimentEvent[]>([]);
  const [artifactTurns, setArtifactTurns] = useState<TurnCompletedPayload[]>([]);
  const [isPolling, setIsPolling] = useState(true);
  const [selectedTurnId, setSelectedTurnId] = useState<string | null>(null);
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

  const eventTurns = useMemo(() => turnsFromEvents(events), [events]);
  const turns = artifactTurns.length > 0 ? artifactTurns : eventTurns;
  const userFilteredTurns = useMemo(() => userTurns(turns, selectedUser), [turns, selectedUser]);
  const summary = useMemo(() => summarizeTurns(userFilteredTurns), [userFilteredTurns]);
  const users = useMemo(() => uniqueValues(turns, 'user_id'), [turns]);
  const actionsByPolicy = useMemo(() => policyCounts(userFilteredTurns, 'action_id'), [userFilteredTurns]);
  const responsesByPolicy = useMemo(() => policyCounts(userFilteredTurns, 'response_type'), [userFilteredTurns]);
  const selectedTurn = useMemo(() => {
    if (!selectedTurnId) return userFilteredTurns.at(-1) ?? null;
    return userFilteredTurns.find((turn) => turnId(turn) === selectedTurnId) ?? userFilteredTurns.at(-1) ?? null;
  }, [userFilteredTurns, selectedTurnId]);

  async function handleArtifactUpload(file: File | null) {
    if (!file) return;
    const text = await file.text();
    const parsed = JSON.parse(text) as ExperimentArtifact;
    setArtifactTurns(turnsFromArtifact(parsed));
    setIsPolling(false);
    setLastLoaded(`loaded ${file.name}`);
  }

  return (
    <main className="shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Emotiv Learn Observatory</p>
          <h1>Policy experiment console</h1>
          <p className="subtle">
            A quiet control surface for watching Tutor LLM calls, hidden-student transitions, observable rewards, and policy updates.
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
        <Metric label="turn events" value={String(userFilteredTurns.length)} />
        <Metric label="users" value={String(users.length)} />
        <Metric label="policies" value={String(Object.keys(summary).length)} />
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

      <section className="layout">
        <div className="leftColumn">
          <Panel title="Policy Comparison" quiet>
            <PolicyTable summary={summary} turns={userFilteredTurns} />
          </Panel>
          <Panel title="Action Mix">
            <Distribution counts={actionsByPolicy} />
          </Panel>
          <Panel title="Response Mix">
            <Distribution counts={responsesByPolicy} />
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
  return (
    <div className="distribution">
      {policyOrder
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

function TurnDetail({ turn }: { turn: TurnCompletedPayload }) {
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

function turnId(turn: TurnCompletedPayload): string {
  return `${turn.policy_mode}:${turn.user_id}:${turn.turn_index}`;
}

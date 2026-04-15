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
import type { ExperimentArtifact, ExperimentEvent, PolicyMode, TurnCompletedPayload } from './types';

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
  const filteredTurns = useMemo(() => userTurns(turns, selectedUser), [turns, selectedUser]);
  const summary = useMemo(() => summarizeTurns(filteredTurns), [filteredTurns]);
  const users = useMemo(() => uniqueValues(turns, 'user_id'), [turns]);
  const actionsByPolicy = useMemo(() => policyCounts(filteredTurns, 'action_id'), [filteredTurns]);
  const responsesByPolicy = useMemo(() => policyCounts(filteredTurns, 'response_type'), [filteredTurns]);
  const selectedTurn = useMemo(() => {
    if (!selectedTurnId) return filteredTurns.at(-1) ?? null;
    return filteredTurns.find((turn) => turnId(turn) === selectedTurnId) ?? filteredTurns.at(-1) ?? null;
  }, [filteredTurns, selectedTurnId]);

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
        <Metric label="turn events" value={String(filteredTurns.length)} />
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
            <PolicyTable summary={summary} turns={filteredTurns} />
          </Panel>
          <Panel title="Action Mix">
            <Distribution counts={actionsByPolicy} />
          </Panel>
          <Panel title="Response Mix">
            <Distribution counts={responsesByPolicy} />
          </Panel>
        </div>

        <Panel title="Live Turn Stream" className="streamPanel">
          <TurnStream turns={filteredTurns} selectedId={selectedTurn ? turnId(selectedTurn) : null} onSelect={setSelectedTurnId} />
        </Panel>

        <div className="rightColumn">
          <Panel title="Selected Turn">
            {selectedTurn ? <TurnDetail turn={selectedTurn} /> : <EmptyState />}
          </Panel>
          <Panel title="Reward / Oracle Trace">
            <Trace turns={filteredTurns} />
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

function TurnStream({ turns, selectedId, onSelect }: { turns: TurnCompletedPayload[]; selectedId: string | null; onSelect: (id: string) => void }) {
  return (
    <div className="turnList">
      {[...turns].reverse().map((turn) => {
        const id = turnId(turn);
        return (
          <button className={`turnItem ${selectedId === id ? 'selected' : ''}`} key={id} onClick={() => onSelect(id)}>
            <span className="mono">t{turn.turn_index}</span>
            <span>{turn.policy_mode}</span>
            <span>{turn.user_id}</span>
            <strong>{turn.action_id}</strong>
            <em>{turn.response_type}</em>
            <span className={turn.reward >= 0 ? 'positive' : 'negative'}>{formatNumber(turn.reward)}</span>
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

function Trace({ turns }: { turns: TurnCompletedPayload[] }) {
  const values = turns.slice(-80);
  if (values.length === 0) return <EmptyState />;
  const maxReward = Math.max(...values.map((turn) => Math.abs(turn.reward)), 1);
  const maxGain = Math.max(...values.map((turn) => turn.oracle_mastery_gain), 0.01);
  return (
    <div className="trace">
      {values.map((turn) => (
        <div className="traceColumn" key={turnId(turn)} title={`${turn.policy_mode} t${turn.turn_index}`}>
          <span style={{ height: `${Math.max((Math.abs(turn.reward) / maxReward) * 72, 2)}px` }} className={turn.reward >= 0 ? 'rewardBar' : 'lossBar'} />
          <span style={{ height: `${Math.max((turn.oracle_mastery_gain / maxGain) * 72, 2)}px` }} className="gainBar" />
        </div>
      ))}
    </div>
  );
}

function EmptyState() {
  return <div className="empty">No experiment events yet. Start a run or load an artifact.</div>;
}

function turnId(turn: TurnCompletedPayload): string {
  return `${turn.policy_mode}:${turn.user_id}:${turn.turn_index}`;
}

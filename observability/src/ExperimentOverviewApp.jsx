import React, { useEffect, useMemo, useState } from "react";
import { Activity, ArrowRight, Brain, Gauge, Sparkles, TrendingUp } from "lucide-react";
import {
  actionColor,
  averageByTurn,
  buildSparklinePoints,
  computeUserUpliftRows,
  countsToSortedEntries,
  dominantAction,
  formatNumber,
  loadSystem3Comparison,
} from "./experimentData";
import "./experiment-renders.css";

function MetricCard({ label, value, note, icon: Icon }) {
  return (
    <article className="render-metric-card">
      <div className="render-metric-label">
        <Icon className="icon-14" />
        <span>{label}</span>
      </div>
      <div className="render-metric-value">{value}</div>
      <div className="render-metric-note">{note}</div>
    </article>
  );
}

function PolicyPanel({ title, summary, tone }) {
  const actions = countsToSortedEntries(summary?.action_counts ?? {}).slice(0, 4);
  return (
    <section className={`render-policy-panel ${tone}`}>
      <div className="render-panel-kicker">{title}</div>
      <div className="render-policy-grid">
        <div>
          <span>average reward</span>
          <strong>{formatNumber(summary?.average_reward)}</strong>
        </div>
        <div>
          <span>goal progress</span>
          <strong>{formatNumber(summary?.average_goal_progress)}</strong>
        </div>
        <div>
          <span>concepts mastered</span>
          <strong>{formatNumber(summary?.average_concepts_mastered, 1)}</strong>
        </div>
        <div>
          <span>dominant action</span>
          <strong>{dominantAction(summary?.action_counts)}</strong>
        </div>
      </div>
      <div className="render-action-row">
        {actions.map(([actionId, count]) => (
          <span key={actionId} className="render-action-pill" style={{ borderColor: actionColor(actionId), color: actionColor(actionId) }}>
            {actionId}
            <strong>{count}</strong>
          </span>
        ))}
      </div>
    </section>
  );
}

function TrendChart({ title, leftSeries, rightSeries, leftLabel, rightLabel, leftColor, rightColor }) {
  const width = 720;
  const height = 250;
  const padding = 26;
  const values = [...leftSeries.map((item) => item.value), ...rightSeries.map((item) => item.value)];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, 1e-6);
  const buildPath = (series) =>
    series
      .map((item, index) => {
        const x = padding + (index / Math.max(series.length - 1, 1)) * (width - padding * 2);
        const y = height - padding - ((item.value - min) / span) * (height - padding * 2);
        return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
      })
      .join(" ");

  return (
    <section className="render-chart-card">
      <div className="render-card-top">
        <div>
          <p className="render-card-kicker">trendline</p>
          <h3>{title}</h3>
        </div>
        <div className="render-legend">
          <span><i style={{ background: leftColor }} />{leftLabel}</span>
          <span><i style={{ background: rightColor }} />{rightLabel}</span>
        </div>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="render-chart-svg">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const y = padding + t * (height - padding * 2);
          return <line key={t} x1={padding} x2={width - padding} y1={y} y2={y} className="render-grid-line" />;
        })}
        <path d={buildPath(leftSeries)} fill="none" stroke={leftColor} strokeWidth="3" strokeLinecap="round" />
        <path d={buildPath(rightSeries)} fill="none" stroke={rightColor} strokeWidth="3" strokeLinecap="round" />
      </svg>
    </section>
  );
}

function UserRow({ row, data }) {
  const profile = data.user_profiles?.[row.userId];
  const personalizedTurns = (data.results?.personalized?.turns ?? []).filter((turn) => turn.user_id === row.userId);
  const baselineTurns = (data.results?.no_policy?.turns ?? []).filter((turn) => turn.user_id === row.userId);
  const personalizedPath = buildSparklinePoints(personalizedTurns.map((turn) => turn.reward ?? 0));
  const baselinePath = buildSparklinePoints(baselineTurns.map((turn) => turn.reward ?? 0));

  return (
    <article className="render-user-row">
      <div className="render-user-copy">
        <p className="render-user-kicker">{row.userId}</p>
        <h4>{profile?.description ?? "Long-run simulator profile"}</h4>
        <div className="render-user-metrics">
          <span>reward delta <strong>{formatNumber(row.rewardDelta)}</strong></span>
          <span>goal delta <strong>{formatNumber(row.goalDelta)}</strong></span>
          <span>best action <strong>{dominantAction(row.personalized?.action_counts)}</strong></span>
        </div>
      </div>
      <div className="render-user-spark">
        <svg viewBox="0 0 220 56" className="render-spark-svg">
          <path d={baselinePath} fill="none" stroke="#c7c7cc" strokeWidth="2.5" strokeLinecap="round" />
          <path d={personalizedPath} fill="none" stroke="#8b5cf6" strokeWidth="2.5" strokeLinecap="round" />
        </svg>
        <div className="render-legend render-legend-tight">
          <span><i style={{ background: "#8b5cf6" }} />personalized</span>
          <span><i style={{ background: "#c7c7cc" }} />no policy</span>
        </div>
      </div>
    </article>
  );
}

export default function ExperimentOverviewApp() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    loadSystem3Comparison()
      .then((payload) => {
        if (!cancelled) setData(payload);
      })
      .catch((err) => {
        if (!cancelled) setError(String(err?.message ?? err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const rewardSeries = useMemo(() => {
    if (!data) return { personalized: [], baseline: [] };
    return {
      personalized: averageByTurn(data.results.personalized.turns, (turn) => turn.reward ?? 0),
      baseline: averageByTurn(data.results.no_policy.turns, (turn) => turn.reward ?? 0),
    };
  }, [data]);

  const goalSeries = useMemo(() => {
    if (!data) return { personalized: [], baseline: [] };
    return {
      personalized: averageByTurn(data.results.personalized.turns, (turn) => turn.outcome?.goal_progress ?? 0),
      baseline: averageByTurn(data.results.no_policy.turns, (turn) => turn.outcome?.goal_progress ?? 0),
    };
  }, [data]);

  const upliftRows = useMemo(() => (data ? computeUserUpliftRows(data) : []), [data]);

  if (error) {
    return <main className="render-empty">{error}</main>;
  }
  if (!data) {
    return <main className="render-empty">Loading experiment overview...</main>;
  }

  const rewardDelta = (data.summary?.personalized?.average_reward ?? 0) - (data.summary?.no_policy?.average_reward ?? 0);
  const goalDelta = (data.summary?.personalized?.average_goal_progress ?? 0) - (data.summary?.no_policy?.average_goal_progress ?? 0);

  return (
    <main className="render-shell">
      <header className="render-hero">
        <p className="render-hero-kicker">Long-run simulator render</p>
        <h1 className="render-hero-title">Sena</h1>
        <p className="render-hero-subtitle">
          Static overview of the long-running System 3 tutor trials. Personalized policy is compared
          against the simulator baseline across {data.turns} turns and {Object.keys(data.user_profiles ?? {}).length} learners.
        </p>
      </header>

      <section className="render-metric-strip">
        <MetricCard label="trial length" value={`${data.turns} turns`} note="per policy run" icon={Activity} />
        <MetricCard label="learners" value={`${Object.keys(data.user_profiles ?? {}).length}`} note="simulator profiles" icon={Brain} />
        <MetricCard label="reward uplift" value={formatNumber(rewardDelta)} note="personalized minus baseline" icon={TrendingUp} />
        <MetricCard label="goal uplift" value={formatNumber(goalDelta)} note="average final progress delta" icon={Gauge} />
      </section>

      <section className="render-policy-columns">
        <PolicyPanel title="Personalized policy" summary={data.summary?.personalized} tone="render-policy-primary" />
        <div className="render-policy-divider"><ArrowRight className="icon-14" /></div>
        <PolicyPanel title="No-policy baseline" summary={data.summary?.no_policy} tone="render-policy-secondary" />
      </section>

      <section className="render-chart-grid">
        <TrendChart
          title="Average reward by turn"
          leftSeries={rewardSeries.personalized}
          rightSeries={rewardSeries.baseline}
          leftLabel="personalized"
          rightLabel="no policy"
          leftColor="#8b5cf6"
          rightColor="#c7c7cc"
        />
        <TrendChart
          title="Average goal progress by turn"
          leftSeries={goalSeries.personalized}
          rightSeries={goalSeries.baseline}
          leftLabel="personalized"
          rightLabel="no policy"
          leftColor="#3b82f6"
          rightColor="#d1d5db"
        />
      </section>

      <section className="render-user-grid-card">
        <div className="render-card-top">
          <div>
            <p className="render-card-kicker">learner divergence</p>
            <h3>Per-user long-run outcomes</h3>
          </div>
          <div className="render-inline-note">
            <Sparkles className="icon-14" />
            <span>each sparkline shows reward trajectory across all 50 turns</span>
          </div>
        </div>
        <div className="render-user-grid">
          {upliftRows.map((row) => (
            <UserRow key={row.userId} row={row} data={data} />
          ))}
        </div>
      </section>
    </main>
  );
}

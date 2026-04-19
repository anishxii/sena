import React, { useEffect, useMemo, useState } from "react";
import { ArrowRight, Brain, Sparkles, TrendingUp } from "lucide-react";
import {
  actionColor,
  computeUserUpliftRows,
  countsToSortedEntries,
  formatNumber,
  loadSystem3Comparison,
  sampleTurnsForUser,
} from "./experimentData";
import "./experiment-renders.css";

function ActionStack({ counts, muted }) {
  const top = countsToSortedEntries(counts).slice(0, 5);
  return (
    <div className={`spotlight-action-stack ${muted ? "muted" : ""}`}>
      {top.map(([actionId, count]) => (
        <div key={actionId} className="spotlight-action-row">
          <span>{actionId}</span>
          <div className="spotlight-action-bar">
            <i style={{ width: `${Math.min(100, count / 0.5)}%`, background: actionColor(actionId) }} />
          </div>
          <strong>{count}</strong>
        </div>
      ))}
    </div>
  );
}

function TurnCard({ pair }) {
  const rewardDelta = (pair.personalized.reward ?? 0) - (pair.baseline.reward ?? 0);
  return (
    <article className="spotlight-turn-card">
      <div className="spotlight-turn-top">
        <span>turn {pair.timestamp}</span>
        <strong>{rewardDelta >= 0 ? "+" : ""}{formatNumber(rewardDelta)}</strong>
      </div>
      <div className="spotlight-turn-columns">
        <section>
          <p className="spotlight-turn-label">personalized</p>
          <div className="spotlight-token" style={{ borderColor: actionColor(pair.personalized.selected_action), color: actionColor(pair.personalized.selected_action) }}>
            {pair.personalized.selected_action}
          </div>
          <p>{pair.personalized.outcome?.tutor_message}</p>
          <small>{pair.personalized.outcome?.student_message}</small>
        </section>
        <section>
          <p className="spotlight-turn-label">baseline</p>
          <div className="spotlight-token spotlight-token-muted">{pair.baseline.selected_action}</div>
          <p>{pair.baseline.outcome?.tutor_message}</p>
          <small>{pair.baseline.outcome?.student_message}</small>
        </section>
      </div>
    </article>
  );
}

function SpotlightCard({ row, data }) {
  const profile = data.user_profiles?.[row.userId];
  const samples = sampleTurnsForUser(data, row.userId);
  return (
    <section className="spotlight-card">
      <div className="spotlight-card-header">
        <div>
          <p className="render-card-kicker">{row.userId}</p>
          <h3>{profile?.description ?? "Simulator learner profile"}</h3>
        </div>
        <div className="spotlight-summary">
          <span><TrendingUp className="icon-14" /> reward delta <strong>{formatNumber(row.rewardDelta)}</strong></span>
          <span><Brain className="icon-14" /> goal delta <strong>{formatNumber(row.goalDelta)}</strong></span>
        </div>
      </div>

      <div className="spotlight-policy-compare">
        <div>
          <p className="spotlight-turn-label">personalized action mix</p>
          <ActionStack counts={row.personalized?.action_counts} />
        </div>
        <div className="spotlight-arrow"><ArrowRight className="icon-14" /></div>
        <div>
          <p className="spotlight-turn-label">baseline action mix</p>
          <ActionStack counts={row.baseline?.action_counts} muted />
        </div>
      </div>

      <div className="spotlight-turn-grid">
        {samples.map((pair) => (
          <TurnCard key={`${row.userId}-${pair.timestamp}`} pair={pair} />
        ))}
      </div>
    </section>
  );
}

export default function ExperimentSpotlightApp() {
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

  const spotlightRows = useMemo(() => {
    if (!data) return [];
    return computeUserUpliftRows(data).slice(0, 3);
  }, [data]);

  if (error) {
    return <main className="render-empty">{error}</main>;
  }
  if (!data) {
    return <main className="render-empty">Loading trial spotlight...</main>;
  }

  return (
    <main className="render-shell">
      <header className="render-hero">
        <p className="render-hero-kicker">Trial spotlight render</p>
        <h1 className="render-hero-title">Sena</h1>
        <p className="render-hero-subtitle">
          User-level divergence snapshots from the long-running System 3 simulator jobs.
          Each card shows where the personalized tutor departs from the fixed baseline and how that changes the trial trajectory.
        </p>
      </header>

      <section className="spotlight-intro-card">
        <div className="render-inline-note">
          <Sparkles className="icon-14" />
          <span>These are static presentation renders generated from the same simulator output file used by the comparison viewer.</span>
        </div>
      </section>

      <div className="spotlight-stack">
        {spotlightRows.map((row) => (
          <SpotlightCard key={row.userId} row={row} data={data} />
        ))}
      </div>
    </main>
  );
}

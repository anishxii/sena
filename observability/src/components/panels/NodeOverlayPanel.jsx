import React from "react";
import { createPortal } from "react-dom";
import { X } from "lucide-react";
import {
  Bar,
  BarChart,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { formatNumber } from "../../data";
import {
  buildCognitiveStateSeries,
  buildOutcomeSummary,
  buildPolicyDecisionSummary,
  buildRawObservationSeries,
} from "../../replay/viewModel";

const RAW_SIGNAL_SPECS = [
  {
    key: "workload_estimate",
    label: "Workload estimate",
    color: "rgba(255,255,255,0.92)",
  },
  {
    key: "theta_alpha_ratio",
    label: "Theta/alpha ratio",
    color: "rgba(196,181,253,0.92)",
  },
  {
    key: "progress_signal",
    label: "Progress signal",
    color: "rgba(134,239,172,0.92)",
  },
];

const COGNITIVE_SIGNAL_SPECS = {
  overload_risk: {
    label: "Overload risk",
    color: "rgba(248,113,113,0.92)",
  },
  future_lapse_risk: {
    label: "Future lapse risk",
    color: "rgba(251,191,36,0.92)",
  },
  attention_stability: {
    label: "Attention stability",
    color: "rgba(96,165,250,0.92)",
  },
  engagement: {
    label: "Engagement",
    color: "rgba(52,211,153,0.92)",
  },
  confidence: {
    label: "Confidence",
    color: "rgba(244,114,182,0.92)",
  },
  needs_structure: {
    label: "Needs structure",
    color: "rgba(148,163,184,0.92)",
  },
  readiness_for_depth: {
    label: "Readiness for depth",
    color: "rgba(196,181,253,0.92)",
  },
};

const POLICY_COLORS = [
  "rgba(196,181,253,0.92)",
  "rgba(96,165,250,0.92)",
  "rgba(52,211,153,0.92)",
  "rgba(251,191,36,0.92)",
  "rgba(248,113,113,0.92)",
];

function SignalTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;

  return (
    <div className="signal-tooltip">
      <div className="signal-tooltip-label">{label}</div>
      {payload.map((entry) => (
        <div key={entry.dataKey} className="signal-tooltip-row">
          <span className="signal-tooltip-swatch" style={{ background: entry.color }} />
          <span>{entry.name}</span>
          <strong>{Number(entry.value ?? 0).toFixed(3)}</strong>
        </div>
      ))}
    </div>
  );
}

function SignalChart({ data, dataKey, color, label }) {
  if (!data.length) {
    return <div className="sparkline-empty">No data</div>;
  }

  return (
    <div className="signal-chart-shell">
      <ResponsiveContainer width="100%" height={164}>
        <LineChart data={data} margin={{ top: 10, right: 10, bottom: 2, left: -18 }}>
          <XAxis
            dataKey="label"
            stroke="rgba(141,144,152,0.72)"
            tick={{ fill: "rgba(141,144,152,0.72)", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
          />
          <YAxis
            stroke="rgba(141,144,152,0.72)"
            tick={{ fill: "rgba(141,144,152,0.72)", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
            domain={["auto", "auto"]}
            width={34}
          />
          <Tooltip content={<SignalTooltip />} cursor={{ stroke: "rgba(255,255,255,0.08)" }} />
          <Line
            type="monotone"
            dataKey={dataKey}
            name={label}
            stroke={color}
            strokeWidth={2}
            dot={{ r: 3.5, fill: color, stroke: "rgba(8,10,14,0.92)", strokeWidth: 1.25 }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function SignalCard({ label, value, color, data, dataKey }) {
  return (
    <div className="overlay-signal-card">
      <div className="overlay-signal-header">
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
      <SignalChart data={data} dataKey={dataKey} color={color} label={label} />
    </div>
  );
}

function PolicyBreakdownPanel({ replayEvents }) {
  const { barData, pieData, log } = buildPolicyDecisionSummary(replayEvents);

  return (
    <>
      <div className="overlay-panel-copy">
        Decision-time policy traces over the currently visible replay window. The charts summarize what the engine preferred, and the log shows the exact action scores available at each decision event.
      </div>

      <div className="policy-chart-grid">
        <div className="overlay-signal-card">
          <div className="overlay-signal-header">
            <span>Average action score</span>
            <strong>{barData.length} actions</strong>
          </div>
          <div className="signal-chart-shell">
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={barData} margin={{ top: 10, right: 10, left: -22, bottom: 2 }}>
                <XAxis
                  dataKey="actionId"
                  stroke="rgba(141,144,152,0.72)"
                  tick={{ fill: "rgba(141,144,152,0.72)", fontSize: 10 }}
                  tickLine={false}
                  axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
                />
                <YAxis
                  stroke="rgba(141,144,152,0.72)"
                  tick={{ fill: "rgba(141,144,152,0.72)", fontSize: 11 }}
                  tickLine={false}
                  axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
                  width={34}
                />
                <Tooltip
                  formatter={(value) => formatNumber(value)}
                  cursor={{ fill: "rgba(255,255,255,0.03)" }}
                  contentStyle={{
                    border: "1px solid rgba(255,255,255,0.08)",
                    borderRadius: 14,
                    background: "rgba(12,13,18,0.96)",
                  }}
                />
                <Bar dataKey="averageScore" radius={[6, 6, 0, 0]}>
                  {barData.map((entry, index) => (
                    <Cell key={entry.actionId} fill={POLICY_COLORS[index % POLICY_COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="overlay-signal-card">
          <div className="overlay-signal-header">
            <span>Selected action share</span>
            <strong>{log.length} scored turns</strong>
          </div>
          <div className="signal-chart-shell">
            <ResponsiveContainer width="100%" height={180}>
              <PieChart>
                <Tooltip
                  formatter={(value) => `${value}`}
                  contentStyle={{
                    border: "1px solid rgba(255,255,255,0.08)",
                    borderRadius: 14,
                    background: "rgba(12,13,18,0.96)",
                  }}
                />
                <Pie
                  data={pieData}
                  dataKey="count"
                  nameKey="actionId"
                  innerRadius={44}
                  outerRadius={72}
                  paddingAngle={3}
                >
                  {pieData.map((entry, index) => (
                    <Cell key={entry.actionId} fill={POLICY_COLORS[index % POLICY_COLORS.length]} />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="policy-log-card">
        <div className="overlay-signal-header">
          <span>Action log by event</span>
          <strong>{log.length} entries</strong>
        </div>
        <div className="policy-log-list">
          {log.map((entry) => (
            <div key={entry.label} className="policy-log-entry">
              <div className="policy-log-entry-top">
                <span>{entry.label}</span>
                <strong>{entry.selectedAction}</strong>
              </div>
              <div className="policy-log-entry-score">
                selected score {formatNumber(entry.selectedScore)}
              </div>
              <div className="policy-log-score-list">
                {entry.scores.map((scoreRow) => (
                  <div key={`${entry.label}-${scoreRow.actionId}`} className="policy-log-score-row">
                    <span>{scoreRow.actionId}</span>
                    <strong>{formatNumber(scoreRow.score)}</strong>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}

function OutcomePanel({ replayEvents }) {
  const { series, responseBreakdown, log } = buildOutcomeSummary(replayEvents);
  const latest = series[series.length - 1] ?? {};

  return (
    <>
      <div className="overlay-panel-copy">
        Observed learner response and reward signals over the visible replay window. This is the downstream feedback surface the system uses to measure whether the intervention helped.
      </div>

      <div className="policy-chart-grid">
        <SignalCard
          label="Reward"
          value={latest.reward == null ? "n/a" : formatNumber(latest.reward)}
          color="rgba(196,181,253,0.92)"
          data={series}
          dataKey="reward"
        />
        <SignalCard
          label="Comprehension score"
          value={latest.comprehension_score == null ? "n/a" : formatNumber(latest.comprehension_score)}
          color="rgba(52,211,153,0.92)"
          data={series}
          dataKey="comprehension_score"
        />
        <SignalCard
          label="Confusion score"
          value={latest.confusion_score == null ? "n/a" : formatNumber(latest.confusion_score)}
          color="rgba(248,113,113,0.92)"
          data={series}
          dataKey="confusion_score"
        />
        <div className="overlay-signal-card">
          <div className="overlay-signal-header">
            <span>Response type share</span>
            <strong>{responseBreakdown.length} types</strong>
          </div>
          <div className="signal-chart-shell">
            <ResponsiveContainer width="100%" height={180}>
              <PieChart>
                <Tooltip
                  formatter={(value) => `${value}`}
                  contentStyle={{
                    border: "1px solid rgba(255,255,255,0.08)",
                    borderRadius: 14,
                    background: "rgba(12,13,18,0.96)",
                  }}
                />
                <Pie
                  data={responseBreakdown}
                  dataKey="count"
                  nameKey="responseType"
                  innerRadius={44}
                  outerRadius={72}
                  paddingAngle={3}
                >
                  {responseBreakdown.map((entry, index) => (
                    <Cell key={entry.responseType} fill={POLICY_COLORS[index % POLICY_COLORS.length]} />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="policy-log-card">
        <div className="overlay-signal-header">
          <span>Outcome log by turn</span>
          <strong>{log.length} entries</strong>
        </div>
        <div className="policy-log-list">
          {log.map((entry) => (
            <div key={entry.label} className="policy-log-entry">
              <div className="policy-log-entry-top">
                <span>{entry.label}</span>
                <strong>{entry.response_type}</strong>
              </div>
              <div className="policy-log-entry-score">
                reward {formatNumber(entry.reward)} · checkpoint {entry.checkpointLabel}
              </div>
              <div className="policy-log-score-list">
                <div className="policy-log-score-row">
                  <span>comprehension</span>
                  <strong>{formatNumber(entry.comprehension_score)}</strong>
                </div>
                <div className="policy-log-score-row">
                  <span>confusion</span>
                  <strong>{formatNumber(entry.confusion_score)}</strong>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}

function RawObservationPanel({ replayEvents }) {
  const series = buildRawObservationSeries(replayEvents);
  const latest = series[series.length - 1] ?? {};

  return (
    <>
      <div className="overlay-panel-copy">
        Raw biosignal and behavioral inputs over replay time. This panel is temporal by design, so every metric here shows how the observation evolves across turns.
      </div>
      <div className="overlay-signal-grid">
        {RAW_SIGNAL_SPECS.map((signal) => (
          <SignalCard
            key={signal.key}
            label={signal.label}
            value={latest[signal.key]?.toFixed?.(3) ?? "n/a"}
            color={signal.color}
            data={series}
            dataKey={signal.key}
          />
        ))}
      </div>
    </>
  );
}

function CognitiveStatePanel({ replayEvents }) {
  const series = buildCognitiveStateSeries(replayEvents);
  const latest = series[series.length - 1] ?? {};
  const latestNames = latest.names ?? Object.keys(COGNITIVE_SIGNAL_SPECS);

  return (
    <>
      <div className="overlay-panel-copy">
        Derived cognitive state features from the middleware translation layer. These are the signals the policy actually consumes at decision time.
      </div>
      <div className="overlay-signal-grid">
        {latestNames.map((signalKey) => {
          const spec = COGNITIVE_SIGNAL_SPECS[signalKey] ?? {
            label: signalKey.replaceAll("_", " "),
            color: "rgba(255,255,255,0.92)",
          };

          return (
            <SignalCard
              key={signalKey}
              label={spec.label}
              value={latest[signalKey]?.toFixed?.(3) ?? "n/a"}
              color={spec.color}
              data={series}
              dataKey={signalKey}
            />
          );
        })}
      </div>
    </>
  );
}

export function NodeOverlayPanel({ nodeId, replayEvents, canvasView, onClose }) {
  const title =
    nodeId === "rawObservation"
      ? "Raw Observation"
      : nodeId === "cognitiveState"
        ? "Cognitive State"
        : nodeId === "policyDecision"
          ? "Policy Decision"
          : "Observed Outcome";

  const panel = (
    <>
      <div className="overlay-backdrop" onClick={onClose} />
      <aside className="overlay-panel">
        <div className="overlay-panel-header">
          <div>
            <h2 className="overlay-panel-title">{title}</h2>
          </div>
          <button type="button" className="overlay-panel-close" onClick={onClose} aria-label="Close panel">
            <X className="icon-16" />
          </button>
        </div>

        {nodeId === "rawObservation" ? (
          <RawObservationPanel replayEvents={replayEvents} canvasView={canvasView} />
        ) : nodeId === "cognitiveState" ? (
          <CognitiveStatePanel replayEvents={replayEvents} />
        ) : nodeId === "policyDecision" ? (
          <PolicyBreakdownPanel replayEvents={replayEvents} />
        ) : (
          <OutcomePanel replayEvents={replayEvents} />
        )}
      </aside>
    </>
  );

  if (typeof document === "undefined") {
    return panel;
  }

  return createPortal(panel, document.body);
}

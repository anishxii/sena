export async function loadSystem3Comparison() {
  const response = await fetch("/system3b_comparison.json", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Could not load /system3b_comparison.json (${response.status})`);
  }
  return response.json();
}

export function formatNumber(value, digits = 2) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return value.toFixed(digits);
}

export function formatPercent(value, digits = 0) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return `${(value * 100).toFixed(digits)}%`;
}

export function averageByTurn(turns, selector) {
  const grouped = new Map();
  turns.forEach((turn) => {
    const bucket = grouped.get(turn.timestamp) ?? [];
    bucket.push(selector(turn));
    grouped.set(turn.timestamp, bucket);
  });
  return [...grouped.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([timestamp, values]) => ({
      timestamp,
      value: values.reduce((sum, current) => sum + current, 0) / values.length,
    }));
}

export function countsToSortedEntries(counts = {}) {
  return Object.entries(counts).sort((a, b) => b[1] - a[1]);
}

export function dominantAction(counts = {}) {
  return countsToSortedEntries(counts)[0]?.[0] ?? "-";
}

export function userSummariesById(data, policy) {
  const rows = data?.results?.[policy]?.users ?? [];
  return Object.fromEntries(rows.map((row) => [row.user_id, row]));
}

export function turnsByUser(data, policy, userId) {
  const turns = data?.results?.[policy]?.turns ?? [];
  return turns.filter((turn) => turn.user_id === userId);
}

export function computeUserUpliftRows(data) {
  const personalized = userSummariesById(data, "personalized");
  const baseline = userSummariesById(data, "no_policy");
  return Object.keys(personalized)
    .map((userId) => {
      const p = personalized[userId];
      const b = baseline[userId];
      return {
        userId,
        personalized: p,
        baseline: b,
        rewardDelta: (p?.reward_average ?? 0) - (b?.reward_average ?? 0),
        goalDelta: (p?.goal_progress_final ?? 0) - (b?.goal_progress_final ?? 0),
        conceptsDelta: (p?.concepts_mastered ?? 0) - (b?.concepts_mastered ?? 0),
      };
    })
    .sort((a, b) => b.rewardDelta - a.rewardDelta);
}

export function sampleTurnsForUser(data, userId) {
  const personalized = turnsByUser(data, "personalized", userId);
  const baseline = turnsByUser(data, "no_policy", userId);
  const paired = personalized
    .map((turn) => [turn, baseline.find((candidate) => candidate.timestamp === turn.timestamp)])
    .filter((pair) => pair[1]);
  if (!paired.length) return [];

  const divergent = paired.filter(([left, right]) => {
    const rewardDelta = Math.abs((left.reward ?? 0) - (right.reward ?? 0));
    const goalDelta = Math.abs((left.outcome?.goal_progress ?? 0) - (right.outcome?.goal_progress ?? 0));
    return left.selected_action !== right.selected_action || rewardDelta >= 0.08 || goalDelta >= 0.05;
  });
  const source = divergent.length >= 3 ? divergent : paired;
  const indices = [0, Math.floor(source.length / 2), source.length - 1];
  const unique = [...new Set(indices)].map((index) => source[index]).filter(Boolean);
  return unique.map(([personalizedTurn, baselineTurn]) => ({
    timestamp: personalizedTurn.timestamp,
    personalized: personalizedTurn,
    baseline: baselineTurn,
  }));
}

export function buildSparklinePoints(values, width = 220, height = 56, padding = 6) {
  if (!values.length) return "";
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, 1e-6);
  return values
    .map((value, index) => {
      const x = padding + (index / Math.max(values.length - 1, 1)) * (width - padding * 2);
      const y = height - padding - ((value - min) / span) * (height - padding * 2);
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

export function actionColor(actionId) {
  const palette = {
    no_change: "#9ca3af",
    simplify: "#34c759",
    deepen: "#8b5cf6",
    summarize: "#64748b",
    highlight_key_points: "#3b82f6",
    worked_example: "#f59e0b",
    analogy: "#ef4444",
    step_by_step: "#0ea5e9",
  };
  return palette[actionId] ?? "#1d1d1f";
}

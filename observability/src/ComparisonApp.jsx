import React, { useEffect, useMemo, useState } from 'react';

function formatNumber(value, digits = 2) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '-';
  return value.toFixed(digits);
}

function truncate(text, maxLength) {
  if (!text || typeof text !== 'string') return '-';
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength) + '...';
}

function formatPercent(value) {
  if (value === null || value === undefined) return '-';
  return `${(value * 100).toFixed(0)}%`;
}

function calculateRollingAverage(turns, field, windowSize) {
  return turns.map((turn, i) => {
    const start = Math.max(0, i - Math.floor(windowSize / 2));
    const end = Math.min(turns.length, i + Math.ceil(windowSize / 2));
    const window = turns.slice(start, end);
    const avg = window.reduce((sum, t) => sum + (t[field] || 0), 0) / window.length;
    return { timestamp: turn.timestamp, value: avg };
  });
}

function scoreToColor(score) {
  if (score > 0) {
    const intensity = Math.min(score * 70, 70);
    return `hsl(120, 60%, ${90 - intensity}%)`;
  } else if (score < 0) {
    const intensity = Math.min(-score * 70, 70);
    return `hsl(0, 60%, ${90 - intensity}%)`;
  }
  return '#f5f5f5';
}

function dominantAction(actionCounts = {}) {
  const entries = Object.entries(actionCounts);
  if (!entries.length) return '-';
  entries.sort((a, b) => b[1] - a[1]);
  return entries[0][0];
}

function representativeTimestamps(personalizedRows, noPolicyRows, userId) {
  const pRows = personalizedRows.filter((row) => row.user_id === userId);
  const nRows = noPolicyRows.filter((row) => row.user_id === userId);
  const paired = pRows
    .map((row) => [row, nRows.find((candidate) => candidate.timestamp === row.timestamp)])
    .filter((pair) => pair[1]);

  const candidateTimestamps = paired
    .filter(([pRow, nRow]) =>
      pRow.selected_action !== nRow.selected_action ||
      Math.abs((pRow.reward ?? 0) - (nRow.reward ?? 0)) >= 0.08 ||
      Math.abs((pRow.outcome?.goal_progress ?? 0) - (nRow.outcome?.goal_progress ?? 0)) >= 0.05,
    )
    .map(([pRow]) => pRow.timestamp);

  const allTimestamps = paired.map(([pRow]) => pRow.timestamp);
  const selected = [];
  const source = candidateTimestamps.length >= 3 ? candidateTimestamps : allTimestamps;

  if (source.length > 0) selected.push(source[0]);
  if (source.length > 2) selected.push(source[Math.floor(source.length / 2)]);
  if (source.length > 1) selected.push(source[source.length - 1]);

  return [...new Set(selected)].slice(0, 3);
}

function turnsForUser(rows, userId, timestamps) {
  const timestampSet = new Set(timestamps);
  return rows.filter((row) => row.user_id === userId && timestampSet.has(row.timestamp));
}

export default function ComparisonApp() {
  const [artifact, setArtifact] = useState(null);
  const [error, setError] = useState(null);
  const [selectedUser, setSelectedUser] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function loadArtifact() {
      try {
        const response = await fetch('/system3_comparison.json', { cache: 'no-store' });
        if (!response.ok) {
          throw new Error(`Could not load /system3_comparison.json (${response.status})`);
        }
        const parsed = await response.json();
        if (!cancelled) {
          setArtifact(parsed);
          const firstUser = parsed.results?.personalized?.users?.[0]?.user_id ?? null;
          setSelectedUser(firstUser);
        }
      } catch (err) {
        if (!cancelled) {
          setError(String(err?.message ?? err));
        }
      }
    }

    void loadArtifact();
    return () => {
      cancelled = true;
    };
  }, []);

  const personalizedUsers = artifact?.results?.personalized?.users ?? [];
  const noPolicyUsers = artifact?.results?.no_policy?.users ?? [];
  const userIds = useMemo(() => personalizedUsers.map((row) => row.user_id), [personalizedUsers]);
  const activeUser = selectedUser ?? userIds[0] ?? null;
  const personalizedSummary = personalizedUsers.find((row) => row.user_id === activeUser);
  const noPolicySummary = noPolicyUsers.find((row) => row.user_id === activeUser);
  const highlightedTimestamps = representativeTimestamps(
    artifact?.results?.personalized?.turns ?? [],
    artifact?.results?.no_policy?.turns ?? [],
    activeUser,
  );
  const personalizedTurns = turnsForUser(
    artifact?.results?.personalized?.turns ?? [],
    activeUser,
    highlightedTimestamps,
  );
  const noPolicyTurns = turnsForUser(
    artifact?.results?.no_policy?.turns ?? [],
    activeUser,
    highlightedTimestamps,
  );

  if (error) {
    return (
      <main className="shell">
        <section className="empty">
          <p>{error}</p>
          <p>Generate <code>system3_comparison.json</code> before opening this viewer.</p>
        </section>
      </main>
    );
  }

  if (!artifact) {
    return (
      <main className="shell">
        <section className="empty">
          <p>Loading comparison...</p>
        </section>
      </main>
    );
  }

  const userProfile = artifact?.user_profiles?.[activeUser];

  return (
    <main className="shell">
      <section className="topGrid">
        <PolicyCard
          title="Personalized"
          summary={artifact.summary?.personalized}
          tone="solid"
        />
        <PolicyCard
          title="No policy"
          summary={artifact.summary?.no_policy}
          tone="outline"
        />
      </section>

      <section className="userSelector">
        {userIds.map((userId) => (
          <button
            key={userId}
            className={userId === activeUser ? 'active' : ''}
            onClick={() => setSelectedUser(userId)}
          >
            {userId}
          </button>
        ))}
      </section>

      {userProfile && (
        <section className="learnerProfileCard">
          <div className="learnerName">{activeUser}</div>
          <div className="learnerDescription">{userProfile.description}</div>
        </section>
      )}

      <section className="chartsGrid">
        <ActionDistributionChart
          personalizedCounts={personalizedSummary?.action_counts}
          noPolicyCounts={noPolicySummary?.action_counts}
        />
        <GoalProgressChart
          personalizedTurns={artifact?.results?.personalized?.turns ?? []}
          noPolicyTurns={artifact?.results?.no_policy?.turns ?? []}
          userId={activeUser}
        />
      </section>

      <LearningCurveChart
        personalizedTurns={artifact?.results?.personalized?.turns ?? []}
        noPolicyTurns={artifact?.results?.no_policy?.turns ?? []}
        userId={activeUser}
      />

      <ActionScoreHeatmap
        turns={artifact?.results?.personalized?.turns ?? []}
        userId={activeUser}
      />

      <section className="comparisonGrid">
        <section className="summaryTable">
          <h2>Per-Learner Comparison</h2>
          <table>
            <thead>
              <tr>
                <th>Policy</th>
                <th>Final goal</th>
                <th>Avg reward</th>
                <th>Concepts</th>
                <th>Checkpoints</th>
                <th>Dominant action</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Personalized</td>
                <td className={personalizedSummary?.goal_progress_final > noPolicySummary?.goal_progress_final ? 'betterMetric' : ''}>
                  {formatNumber(personalizedSummary?.goal_progress_final)}
                </td>
                <td className={personalizedSummary?.reward_average > noPolicySummary?.reward_average ? 'betterMetric' : ''}>
                  {formatNumber(personalizedSummary?.reward_average)}
                </td>
                <td className={personalizedSummary?.concepts_mastered > noPolicySummary?.concepts_mastered ? 'betterMetric' : ''}>
                  {personalizedSummary?.concepts_mastered ?? '-'}
                </td>
                <td className={personalizedSummary?.checkpoint_success_rate > noPolicySummary?.checkpoint_success_rate ? 'betterMetric' : ''}>
                  {formatPercent(personalizedSummary?.checkpoint_success_rate)}
                </td>
                <td>{dominantAction(personalizedSummary?.action_counts)}</td>
              </tr>
              <tr>
                <td>No policy</td>
                <td className={noPolicySummary?.goal_progress_final > personalizedSummary?.goal_progress_final ? 'betterMetric' : ''}>
                  {formatNumber(noPolicySummary?.goal_progress_final)}
                </td>
                <td className={noPolicySummary?.reward_average > personalizedSummary?.reward_average ? 'betterMetric' : ''}>
                  {formatNumber(noPolicySummary?.reward_average)}
                </td>
                <td className={noPolicySummary?.concepts_mastered > personalizedSummary?.concepts_mastered ? 'betterMetric' : ''}>
                  {noPolicySummary?.concepts_mastered ?? '-'}
                </td>
                <td className={noPolicySummary?.checkpoint_success_rate > personalizedSummary?.checkpoint_success_rate ? 'betterMetric' : ''}>
                  {formatPercent(noPolicySummary?.checkpoint_success_rate)}
                </td>
                <td>{dominantAction(noPolicySummary?.action_counts)}</td>
              </tr>
            </tbody>
          </table>
        </section>

        <section className="transcriptGrid">
          <TranscriptCard
            title="Personalized"
            turns={personalizedTurns}
            comparisonTurns={noPolicyTurns}
          />
          <TranscriptCard
            title="No policy"
            turns={noPolicyTurns}
            comparisonTurns={personalizedTurns}
          />
        </section>
      </section>
    </main>
  );
}

function PolicyCard({ title, summary, tone }) {
  return (
    <article className={`policyCard ${tone}`}>
      <div className="policyTitle">{title}</div>
      <div className="policyMetrics">
        <div>
          <span>Avg reward</span>
          <strong>{formatNumber(summary?.average_reward)}</strong>
        </div>
        <div>
          <span>Avg goal progress</span>
          <strong>{formatNumber(summary?.average_goal_progress)}</strong>
        </div>
        <div>
          <span>Concepts mastered</span>
          <strong>{formatNumber(summary?.average_concepts_mastered, 1)}</strong>
        </div>
        <div>
          <span>Checkpoint success</span>
          <strong>{formatPercent(summary?.checkpoint_success_rate)}</strong>
        </div>
        <div>
          <span>Top action</span>
          <strong>{dominantAction(summary?.action_counts)}</strong>
        </div>
      </div>
    </article>
  );
}

function TranscriptCard({ title, turns, comparisonTurns }) {
  return (
    <article className="transcriptCard">
      <div className="transcriptTitle">{title}</div>
      <div className="turnMiniStack">
        {turns.map((turn) => {
          const comparisonTurn = comparisonTurns?.find((t) => t.timestamp === turn.timestamp);
          const actionDiffers = comparisonTurn && comparisonTurn.selected_action !== turn.selected_action;

          return (
            <div className="turnMini" key={`${title}:${turn.user_id}:${turn.timestamp}`}>
              <div className="turnMiniTop">
                <span>Turn {turn.timestamp}</span>
                <span className={`actionBadge ${actionDiffers ? 'actionDiffers' : ''}`}>
                  {turn.selected_action}
                </span>
              </div>
              <p className="tutorLine">{truncate(turn.outcome.tutor_message, 200)}</p>
              <p className="studentLine">{truncate(turn.outcome.student_message, 200)}</p>
              <div className="turnMetrics">
                <span>Reward: {formatNumber(turn.reward)}</span>
                <span>Goal: {formatNumber(turn.outcome.goal_progress)}</span>
              </div>
            </div>
          );
        })}
      </div>
    </article>
  );
}

function ActionDistributionChart({ personalizedCounts, noPolicyCounts }) {
  const allActions = [...new Set([
    ...Object.keys(personalizedCounts || {}),
    ...Object.keys(noPolicyCounts || {}),
  ])].sort();

  const maxCount = Math.max(
    ...allActions.map((action) => Math.max(
      personalizedCounts?.[action] || 0,
      noPolicyCounts?.[action] || 0,
    )),
  );

  return (
    <article className="actionChart">
      <h3>Action Distribution</h3>
      <div className="actionChartBars">
        {allActions.map((action) => {
          const pCount = personalizedCounts?.[action] || 0;
          const nCount = noPolicyCounts?.[action] || 0;
          const pWidth = maxCount > 0 ? (pCount / maxCount) * 100 : 0;
          const nWidth = maxCount > 0 ? (nCount / maxCount) * 100 : 0;

          return (
            <div className="actionRow" key={action}>
              <div className="actionLabel">{action}</div>
              <div className="actionBars">
                <div className="actionBarRow">
                  <span className="policyLabel">P</span>
                  <div className="actionBarTrack">
                    <div className="actionBar personalized" style={{ width: `${pWidth}%` }}>
                      {pCount > 0 && <span>{pCount}</span>}
                    </div>
                  </div>
                </div>
                <div className="actionBarRow">
                  <span className="policyLabel">N</span>
                  <div className="actionBarTrack">
                    <div className="actionBar noPolicy" style={{ width: `${nWidth}%` }}>
                      {nCount > 0 && <span>{nCount}</span>}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </article>
  );
}

function GoalProgressChart({ personalizedTurns, noPolicyTurns, userId }) {
  const pData = personalizedTurns
    .filter((t) => t.user_id === userId)
    .map((t) => ({ timestamp: t.timestamp, goal: t.outcome.goal_progress }))
    .sort((a, b) => a.timestamp - b.timestamp);

  const nData = noPolicyTurns
    .filter((t) => t.user_id === userId)
    .map((t) => ({ timestamp: t.timestamp, goal: t.outcome.goal_progress }))
    .sort((a, b) => a.timestamp - b.timestamp);

  if (pData.length === 0 && nData.length === 0) return null;

  const allTimestamps = [...new Set([...pData.map((d) => d.timestamp), ...nData.map((d) => d.timestamp)])].sort((a, b) => a - b);
  const minTimestamp = Math.min(...allTimestamps);
  const maxTimestamp = Math.max(...allTimestamps);
  const timestampRange = maxTimestamp - minTimestamp || 1;

  const width = 500;
  const height = 180;
  const padding = { top: 20, right: 20, bottom: 30, left: 50 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  const toX = (timestamp) => padding.left + ((timestamp - minTimestamp) / timestampRange) * chartWidth;
  const toY = (goal) => padding.top + chartHeight - goal * chartHeight;

  const pPath = pData.map((d, i) => `${i === 0 ? 'M' : 'L'} ${toX(d.timestamp)} ${toY(d.goal)}`).join(' ');
  const nPath = nData.map((d, i) => `${i === 0 ? 'M' : 'L'} ${toX(d.timestamp)} ${toY(d.goal)}`).join(' ');

  return (
    <article className="goalChart">
      <h3>Goal Progress Over Time</h3>
      <svg width={width} height={height} className="chartSvg">
        <g className="axis">
          <line x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} stroke="#ccc" />
          <line x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} stroke="#ccc" />
          <text x={padding.left - 35} y={padding.top + chartHeight / 2} fontSize="11" fill="#666" transform={`rotate(-90, ${padding.left - 35}, ${padding.top + chartHeight / 2})`}>
            Goal Progress
          </text>
          <text x={padding.left + chartWidth / 2} y={height - 5} fontSize="11" fill="#666" textAnchor="middle">
            Turn Number
          </text>
        </g>
        <g className="gridLines">
          {[0, 0.25, 0.5, 0.75, 1.0].map((goal) => (
            <g key={goal}>
              <line
                x1={padding.left}
                y1={toY(goal)}
                x2={width - padding.right}
                y2={toY(goal)}
                stroke="#ececec"
                strokeDasharray="2,2"
              />
              <text x={padding.left - 10} y={toY(goal) + 4} fontSize="10" fill="#999" textAnchor="end">
                {goal.toFixed(2)}
              </text>
            </g>
          ))}
        </g>
        {pPath && (
          <path d={pPath} fill="none" stroke="#111" strokeWidth="2" />
        )}
        {nPath && (
          <path d={nPath} fill="none" stroke="#999" strokeWidth="2" strokeDasharray="4,4" />
        )}
        <g className="legend">
          <g transform="translate(60, 15)">
            <line x1={0} y1={0} x2={20} y2={0} stroke="#111" strokeWidth="2" />
            <text x={25} y={4} fontSize="11" fill="#666">Personalized</text>
          </g>
          <g transform="translate(170, 15)">
            <line x1={0} y1={0} x2={20} y2={0} stroke="#999" strokeWidth="2" strokeDasharray="4,4" />
            <text x={25} y={4} fontSize="11" fill="#666">No policy</text>
          </g>
        </g>
      </svg>
    </article>
  );
}

function LearningCurveChart({ personalizedTurns, noPolicyTurns, userId }) {
  const pTurns = personalizedTurns
    .filter((t) => t.user_id === userId)
    .sort((a, b) => a.timestamp - b.timestamp);

  const nTurns = noPolicyTurns
    .filter((t) => t.user_id === userId)
    .sort((a, b) => a.timestamp - b.timestamp);

  if (pTurns.length === 0 && nTurns.length === 0) return null;

  const pData = calculateRollingAverage(pTurns, 'reward', 5);
  const nData = calculateRollingAverage(nTurns, 'reward', 5);

  const allTimestamps = [...new Set([...pData.map((d) => d.timestamp), ...nData.map((d) => d.timestamp)])].sort((a, b) => a - b);
  const minTimestamp = Math.min(...allTimestamps);
  const maxTimestamp = Math.max(...allTimestamps);
  const timestampRange = maxTimestamp - minTimestamp || 1;

  const allRewards = [...pData.map((d) => d.value), ...nData.map((d) => d.value)];
  const minReward = Math.min(...allRewards);
  const maxReward = Math.max(...allRewards);
  const rewardRange = maxReward - minReward || 1;

  const width = 1000;
  const height = 200;
  const padding = { top: 20, right: 20, bottom: 30, left: 60 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  const toX = (timestamp) => padding.left + ((timestamp - minTimestamp) / timestampRange) * chartWidth;
  const toY = (reward) => padding.top + chartHeight - ((reward - minReward) / rewardRange) * chartHeight;

  const pPath = pData.map((d, i) => `${i === 0 ? 'M' : 'L'} ${toX(d.timestamp)} ${toY(d.value)}`).join(' ');
  const nPath = nData.map((d, i) => `${i === 0 ? 'M' : 'L'} ${toX(d.timestamp)} ${toY(d.value)}`).join(' ');

  const gridValues = [
    minReward,
    minReward + rewardRange * 0.25,
    minReward + rewardRange * 0.5,
    minReward + rewardRange * 0.75,
    maxReward,
  ];

  return (
    <article className="learningCurve">
      <h3>Learning Curve: Reward Over Time</h3>
      <svg width={width} height={height} className="chartSvg">
        <g className="axis">
          <line x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} stroke="#ccc" />
          <line x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} stroke="#ccc" />
          <text x={padding.left - 45} y={padding.top + chartHeight / 2} fontSize="11" fill="#666" transform={`rotate(-90, ${padding.left - 45}, ${padding.top + chartHeight / 2})`}>
            Average Reward
          </text>
          <text x={padding.left + chartWidth / 2} y={height - 5} fontSize="11" fill="#666" textAnchor="middle">
            Turn Number
          </text>
        </g>
        <g className="gridLines">
          {gridValues.map((reward) => (
            <g key={reward}>
              <line
                x1={padding.left}
                y1={toY(reward)}
                x2={width - padding.right}
                y2={toY(reward)}
                stroke="#ececec"
                strokeDasharray="2,2"
              />
              <text x={padding.left - 10} y={toY(reward) + 4} fontSize="10" fill="#999" textAnchor="end">
                {reward.toFixed(2)}
              </text>
            </g>
          ))}
        </g>
        {pPath && (
          <path d={pPath} fill="none" stroke="#111" strokeWidth="2" />
        )}
        {nPath && (
          <path d={nPath} fill="none" stroke="#999" strokeWidth="2" strokeDasharray="4,4" />
        )}
        <g className="legend">
          <g transform="translate(80, 15)">
            <line x1={0} y1={0} x2={20} y2={0} stroke="#111" strokeWidth="2" />
            <text x={25} y={4} fontSize="11" fill="#666">Personalized (5-turn avg)</text>
          </g>
          <g transform="translate(270, 15)">
            <line x1={0} y1={0} x2={20} y2={0} stroke="#999" strokeWidth="2" strokeDasharray="4,4" />
            <text x={25} y={4} fontSize="11" fill="#666">No policy (5-turn avg)</text>
          </g>
        </g>
      </svg>
    </article>
  );
}

function ActionScoreHeatmap({ turns, userId }) {
  const userTurns = turns
    .filter((t) => t.user_id === userId)
    .sort((a, b) => a.timestamp - b.timestamp);

  if (userTurns.length === 0) return null;

  const sampleTurns = [1, 10, 20, 30, 40, 50];
  const firstTurn = userTurns[0];
  if (!firstTurn || !firstTurn.action_scores) return null;

  const actions = Object.keys(firstTurn.action_scores).sort();

  return (
    <article className="heatmapContainer">
      <h3>Action Score Evolution (Personalized Policy)</h3>
      <p style={{ fontSize: '12px', color: '#666', marginBottom: '12px' }}>
        How the engine learns to score each action over time. Green = preferred, Red = avoided, Gray = neutral.
      </p>
      <table className="scoreHeatmap">
        <thead>
          <tr>
            <th>Action</th>
            {sampleTurns.map((turnNum) => (
              <th key={turnNum}>Turn {turnNum}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {actions.map((action) => (
            <tr key={action}>
              <td>{action}</td>
              {sampleTurns.map((turnNum) => {
                const turn = userTurns.find((t) => t.timestamp === turnNum);
                const score = turn?.action_scores?.[action];
                if (score === undefined || score === null) {
                  return <td key={turnNum}>-</td>;
                }
                const bgColor = scoreToColor(score);
                return (
                  <td key={turnNum} style={{ backgroundColor: bgColor }}>
                    {score.toFixed(2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </article>
  );
}

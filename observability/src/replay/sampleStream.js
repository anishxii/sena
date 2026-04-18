import sampleStreamRaw from "../../../artifacts/replays/sample_stream.jsonl?raw";

function parseJsonl(rawText) {
  return rawText
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

export function loadSampleStreamEvents() {
  const baseEvents = parseJsonl(sampleStreamRaw);
  return augmentWithSyntheticTurns(baseEvents);
}

export function listSessions(events) {
  return [...new Set(events.map((event) => event.session_id))];
}

export function getSessionEvents(events, sessionId) {
  return events
    .filter((event) => event.session_id === sessionId)
    .sort((left, right) => {
      if (left.turn_index !== right.turn_index) {
        return left.turn_index - right.turn_index;
      }
      return left.event_index - right.event_index;
    });
}

export function getReplayWindow(events, activeEventIndex) {
  const clampedIndex = Math.max(0, Math.min(activeEventIndex, events.length - 1));
  const visibleEvents = events.slice(0, clampedIndex + 1);
  const currentEvent = visibleEvents[visibleEvents.length - 1] ?? null;
  return { clampedIndex, visibleEvents, currentEvent };
}

function cloneEvent(event) {
  return JSON.parse(JSON.stringify(event));
}

function augmentWithSyntheticTurns(events) {
  const sessionStarted = events.find((event) => event.event_type === "session.started");
  const baseTurnEvents = events.filter((event) => event.turn_index === 0);
  if (!sessionStarted || baseTurnEvents.length === 0) return events;

  const turnProfiles = [
    {
      turnIndex: 1,
      tsOffset: 100,
      eeg: { theta_alpha_ratio: 1.54, workload_estimate: 0.61 },
      behavior: { followup_type: "continue", progress_signal: 0.71 },
      stateValues: [0.61, 0.45, 0.56, 0.66, 0.69, 0.58, 0.51],
      actionScores: { worked_example: 0.68, highlight_key_points: 0.57 },
      selectedAction: "worked_example",
      semanticStyle: "worked_example",
      responseType: "continue",
      reward: 0.53,
      interpretation: { comprehension_score: 0.78, confusion_score: 0.12 },
      rewardTerms: { correctness: 0.35, progress: 0.18 },
    },
    {
      turnIndex: 2,
      tsOffset: 200,
      eeg: { theta_alpha_ratio: 1.18, workload_estimate: 0.43 },
      behavior: { followup_type: "branch", progress_signal: 0.8 },
      stateValues: [0.43, 0.31, 0.73, 0.72, 0.74, 0.42, 0.67],
      actionScores: { worked_example: 0.41, highlight_key_points: 0.44, deepen: 0.79 },
      selectedAction: "deepen",
      semanticStyle: "depth_increase",
      responseType: "branch",
      reward: 0.64,
      interpretation: { comprehension_score: 0.84, confusion_score: 0.08 },
      rewardTerms: { correctness: 0.36, progress: 0.28 },
    },
    {
      turnIndex: 3,
      tsOffset: 300,
      eeg: { theta_alpha_ratio: 1.37, workload_estimate: 0.56 },
      behavior: { followup_type: "continue", progress_signal: 0.86 },
      stateValues: [0.56, 0.28, 0.77, 0.76, 0.79, 0.37, 0.74],
      actionScores: { worked_example: 0.36, highlight_key_points: 0.47, deepen: 0.83 },
      selectedAction: "deepen",
      semanticStyle: "depth_increase",
      responseType: "continue",
      reward: 0.71,
      interpretation: { comprehension_score: 0.89, confusion_score: 0.05 },
      rewardTerms: { correctness: 0.38, progress: 0.33 },
    },
    {
      turnIndex: 4,
      tsOffset: 400,
      eeg: { theta_alpha_ratio: 1.69, workload_estimate: 0.67 },
      behavior: { followup_type: "clarify", progress_signal: 0.59 },
      stateValues: [0.67, 0.52, 0.49, 0.58, 0.54, 0.71, 0.43],
      actionScores: { worked_example: 0.55, highlight_key_points: 0.74, deepen: 0.42 },
      selectedAction: "highlight_key_points",
      semanticStyle: "salience_increase",
      responseType: "clarify",
      reward: 0.34,
      interpretation: { comprehension_score: 0.57, confusion_score: 0.31 },
      rewardTerms: { correctness: 0.18, progress: 0.16 },
    },
    {
      turnIndex: 5,
      tsOffset: 500,
      eeg: { theta_alpha_ratio: 1.29, workload_estimate: 0.48 },
      behavior: { followup_type: "continue", progress_signal: 0.9 },
      stateValues: [0.48, 0.22, 0.81, 0.8, 0.82, 0.29, 0.8],
      actionScores: { worked_example: 0.33, highlight_key_points: 0.51, deepen: 0.88 },
      selectedAction: "deepen",
      semanticStyle: "depth_increase",
      responseType: "continue",
      reward: 0.79,
      interpretation: { comprehension_score: 0.92, confusion_score: 0.04 },
      rewardTerms: { correctness: 0.4, progress: 0.39 },
    },
  ];

  const augmented = [...events];

  for (const profile of turnProfiles) {
    for (const baseEvent of baseTurnEvents) {
      const event = cloneEvent(baseEvent);
      const newTs = event.ts_ms + profile.tsOffset;
      const newTurnIndex = profile.turnIndex;
      const newTraceId = event.trace_id.replace("turn:0000", `turn:${String(newTurnIndex).padStart(4, "0")}`);

      event.turn_index = newTurnIndex;
      event.trace_id = newTraceId;
      event.ts_ms = newTs;

      switch (event.event_type) {
        case "observation.received":
          event.payload.timestamp = newTs;
          event.payload.payload.eeg = profile.eeg;
          event.payload.payload.behavior = profile.behavior;
          event.summary.primary_metric = newTs;
          break;
        case "state.updated":
          event.payload.timestamp = newTs;
          event.payload.features.values = profile.stateValues;
          break;
        case "action.scored":
          event.payload.timestamp = newTs;
          event.payload.scores = profile.actionScores;
          event.payload.selected_action = profile.selectedAction;
          event.summary.primary_metric = Math.max(...Object.values(profile.actionScores));
          break;
        case "action.selected":
          event.payload.action_id = profile.selectedAction;
          event.summary.headline = `${profile.selectedAction} selected`;
          event.summary.primary_metric = profile.actionScores[profile.selectedAction];
          break;
        case "interaction.emitted":
          event.payload.timestamp = newTs;
          event.payload.action_id = profile.selectedAction;
          event.payload.semantic_effect.style = profile.semanticStyle;
          event.payload.rendering_info.style_label = profile.semanticStyle;
          event.summary.subheadline = profile.selectedAction;
          break;
        case "outcome.received":
          event.payload.timestamp = newTs;
          event.payload.action_id = profile.selectedAction;
          event.payload.payload.response_type = profile.responseType;
          event.summary.primary_metric = newTs;
          event.summary.subheadline = profile.selectedAction;
          break;
        case "outcome.interpreted":
          event.payload.signals = profile.interpretation;
          break;
        case "reward.computed":
          event.payload.timestamp = newTs;
          event.payload.action_id = profile.selectedAction;
          event.payload.reward = profile.reward;
          event.payload.outcome.timestamp = newTs - 1;
          event.payload.outcome.action_id = profile.selectedAction;
          event.payload.outcome.payload.response_type = profile.responseType;
          event.payload.interpreted_outcome.signals = profile.interpretation;
          event.payload.reward_breakdown.terms = profile.rewardTerms;
          event.payload.reward_breakdown.total_reward = profile.reward;
          event.summary.primary_metric = profile.reward;
          event.summary.subheadline = profile.selectedAction;
          break;
        case "turn.committed":
          event.payload.raw_observation.timestamp = newTs - 5;
          event.payload.raw_observation.payload.eeg = profile.eeg;
          event.payload.raw_observation.payload.behavior = profile.behavior;
          event.payload.state.timestamp = newTs - 4;
          event.payload.state.features.values = profile.stateValues;
          event.payload.action_scores.timestamp = newTs - 3;
          event.payload.action_scores.scores = profile.actionScores;
          event.payload.action_scores.selected_action = profile.selectedAction;
          event.payload.action.action_id = profile.selectedAction;
          event.payload.interaction_effect.timestamp = newTs - 2;
          event.payload.interaction_effect.action_id = profile.selectedAction;
          event.payload.interaction_effect.semantic_effect.style = profile.semanticStyle;
          event.payload.interaction_effect.rendering_info.style_label = profile.semanticStyle;
          event.payload.outcome.timestamp = newTs - 1;
          event.payload.outcome.action_id = profile.selectedAction;
          event.payload.outcome.payload.response_type = profile.responseType;
          event.payload.interpreted_outcome.signals = profile.interpretation;
          event.payload.reward_event.timestamp = newTs;
          event.payload.reward_event.action_id = profile.selectedAction;
          event.payload.reward_event.reward = profile.reward;
          event.payload.reward_event.outcome.action_id = profile.selectedAction;
          event.payload.reward_event.outcome.payload.response_type = profile.responseType;
          event.payload.reward_event.interpreted_outcome.signals = profile.interpretation;
          event.payload.reward_event.reward_breakdown.terms = profile.rewardTerms;
          event.payload.reward_event.reward_breakdown.total_reward = profile.reward;
          event.summary.primary_metric = profile.reward;
          event.summary.subheadline = profile.selectedAction;
          break;
        default:
          break;
      }

      augmented.push(event);
    }
  }

  return augmented.sort((left, right) => {
    if (left.turn_index !== right.turn_index) return left.turn_index - right.turn_index;
    return left.event_index - right.event_index;
  });
}

import { formatNumber } from "../data";

const NODE_EVENT_TYPES = {
  rawObservation: ["observation.received"],
  cognitiveState: ["state.updated"],
  policyDecision: ["action.scored", "action.selected"],
  applicationEffect: ["interaction.emitted"],
  outcome: ["outcome.received", "outcome.interpreted", "reward.computed"],
};

function latestEvent(events, eventTypes) {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    if (eventTypes.includes(events[index].event_type)) {
      return events[index];
    }
  }
  return null;
}

function eventSeries(events, eventType) {
  return events.filter((event) => event.event_type === eventType);
}

function getRawObservationView(event) {
  const payload = event?.payload?.payload ?? {};
  const eeg = payload.eeg ?? {};
  return {
    metric: `${formatNumber(eeg.workload_estimate)} load`,
    detail: `theta/alpha ${formatNumber(eeg.theta_alpha_ratio)}`,
  };
}

function getStateView(event) {
  const payload = event?.payload ?? {};
  const featureNames = payload.features?.names ?? [];
  const featureValues = payload.features?.values ?? [];
  const readinessIndex = featureNames.indexOf("readiness_for_depth");
  const readinessValue = readinessIndex >= 0 ? featureValues[readinessIndex] : null;
  return {
    metric: `${featureValues.length} dimensions`,
    detail: `readiness ${formatNumber(readinessValue)}`,
  };
}

function getDecisionView(events) {
  const selected = latestEvent(events, ["action.selected"]);
  const scored = latestEvent(events, ["action.scored"]);
  const selectedAction = selected?.payload?.action_id ?? scored?.payload?.selected_action ?? "n/a";
  const policyType = scored?.payload?.policy_info?.policy_type ?? "n/a";
  return {
    metric: selectedAction,
    detail: `policy ${policyType}`,
  };
}

function getInteractionView(event) {
  const payload = event?.payload ?? {};
  const semanticEffect = payload.semantic_effect ?? {};
  const renderingInfo = payload.rendering_info ?? {};
  const styleLabel = renderingInfo.style_label ?? payload.action_id ?? "n/a";
  return {
    metric: styleLabel,
    detail: `${Object.keys(semanticEffect).length} semantic fields`,
  };
}

function getOutcomeView(events) {
  const reward = latestEvent(events, ["reward.computed"]);
  const interpreted = latestEvent(events, ["outcome.interpreted"]);
  const rewardValue = reward?.payload?.reward;
  const signals = interpreted?.payload?.signals ?? {};
  const responseType = reward?.payload?.outcome?.payload?.response_type ?? "n/a";
  return {
    metric: rewardValue == null ? "reward n/a" : `reward ${formatNumber(rewardValue)}`,
    detail: `${responseType} · ${Object.keys(signals).length} signals`,
  };
}

export function buildCanvasView(visibleEvents) {
  const sessionStarted = visibleEvents.find((event) => event.event_type === "session.started");
  const turnEvents = visibleEvents.filter((event) => event.turn_index >= 0);
  const currentTurn = turnEvents.length ? turnEvents[turnEvents.length - 1].turn_index : 0;
  const currentEvent = turnEvents.length ? turnEvents[turnEvents.length - 1] : sessionStarted;
  const rawObservationEvent = latestEvent(visibleEvents, NODE_EVENT_TYPES.rawObservation);
  const cognitiveStateEvent = latestEvent(visibleEvents, NODE_EVENT_TYPES.cognitiveState);
  const actionScoredEvent = latestEvent(visibleEvents, ["action.scored"]);
  const actionSelectedEvent = latestEvent(visibleEvents, ["action.selected"]);
  const interactionEvent = latestEvent(visibleEvents, NODE_EVENT_TYPES.applicationEffect);
  const outcomeReceivedEvent = latestEvent(visibleEvents, ["outcome.received"]);
  const outcomeInterpretedEvent = latestEvent(visibleEvents, ["outcome.interpreted"]);
  const rewardEvent = latestEvent(visibleEvents, ["reward.computed"]);

  return {
    sessionLabel: sessionStarted?.payload?.user_id ?? "session",
    contextLabel:
      sessionStarted?.payload?.metadata?.topic_id ??
      currentEvent?.summary?.subheadline ??
      "stream replay",
    turnIndex: currentTurn,
    currentEventType: currentEvent?.event_type ?? "session.started",
    currentEvent,
    totalEvents: visibleEvents.length,
    nodePayloads: {
      rawObservation: rawObservationEvent,
      cognitiveState: cognitiveStateEvent,
      policyDecision: actionSelectedEvent ?? actionScoredEvent,
      applicationEffect: interactionEvent,
      outcome: rewardEvent ?? outcomeInterpretedEvent ?? outcomeReceivedEvent,
    },
    nodeViews: {
      rawObservation: getRawObservationView(rawObservationEvent),
      cognitiveState: getStateView(cognitiveStateEvent),
      policyDecision: getDecisionView(visibleEvents),
      applicationEffect: getInteractionView(interactionEvent),
      outcome: getOutcomeView(visibleEvents),
    },
  };
}

export function buildRawObservationSeries(sessionEvents) {
  return eventSeries(sessionEvents, "observation.received").map((event, index) => ({
    index,
    turn: event.turn_index,
    label: `t${event.turn_index}`,
    workload_estimate: event.payload?.payload?.eeg?.workload_estimate ?? 0,
    theta_alpha_ratio: event.payload?.payload?.eeg?.theta_alpha_ratio ?? 0,
    progress_signal: event.payload?.payload?.behavior?.progress_signal ?? 0,
  }));
}

export function buildCognitiveStateSeries(sessionEvents) {
  return eventSeries(sessionEvents, "state.updated").map((event, index) => {
    const names = event.payload?.features?.names ?? [];
    const values = event.payload?.features?.values ?? [];
    const valueFor = (name) => {
      const idx = names.indexOf(name);
      return idx >= 0 ? values[idx] : 0;
    };

    return {
      index,
      turn: event.turn_index,
      label: `t${event.turn_index}`,
      names,
      overload_risk: valueFor("overload_risk"),
      future_lapse_risk: valueFor("future_lapse_risk"),
      attention_stability: valueFor("attention_stability"),
      engagement: valueFor("engagement"),
      confidence: valueFor("confidence"),
      needs_structure: valueFor("needs_structure"),
      readiness_for_depth: valueFor("readiness_for_depth"),
    };
  });
}

export function buildPolicyDecisionSummary(sessionEvents) {
  const scoredEvents = eventSeries(sessionEvents, "action.scored");
  const selectedEvents = eventSeries(sessionEvents, "action.selected");
  const selectedByTurn = new Map(selectedEvents.map((event) => [event.turn_index, event.payload?.action_id ?? "n/a"]));
  const scoreTotals = new Map();
  const scoreCounts = new Map();
  const actionSelections = new Map();

  const log = scoredEvents.map((event) => {
    const scores = event.payload?.scores ?? {};
    const selectedAction =
      event.payload?.selected_action ?? selectedByTurn.get(event.turn_index) ?? "n/a";

    Object.entries(scores).forEach(([actionId, score]) => {
      scoreTotals.set(actionId, (scoreTotals.get(actionId) ?? 0) + Number(score ?? 0));
      scoreCounts.set(actionId, (scoreCounts.get(actionId) ?? 0) + 1);
    });
    actionSelections.set(selectedAction, (actionSelections.get(selectedAction) ?? 0) + 1);

    return {
      turn: event.turn_index,
      label: `t${event.turn_index}`,
      selectedAction,
      selectedScore: Number(scores[selectedAction] ?? 0),
      scores: Object.entries(scores)
        .map(([actionId, score]) => ({ actionId, score: Number(score ?? 0) }))
        .sort((left, right) => right.score - left.score),
    };
  });

  const actions = [...new Set([...scoreTotals.keys(), ...actionSelections.keys()])];
  const barData = actions.map((actionId) => ({
    actionId,
    averageScore:
      (scoreTotals.get(actionId) ?? 0) / Math.max(1, scoreCounts.get(actionId) ?? 0),
  }));
  const pieData = actions.map((actionId) => ({
    actionId,
    count: actionSelections.get(actionId) ?? 0,
  }));

  return { barData, pieData, log };
}

export function buildOutcomeSummary(sessionEvents) {
  const rewardEvents = eventSeries(sessionEvents, "reward.computed");
  const interpretedEvents = eventSeries(sessionEvents, "outcome.interpreted");
  const interpretedByTurn = new Map(
    interpretedEvents.map((event) => [event.turn_index, event.payload?.signals ?? {}])
  );

  const series = rewardEvents.map((event, index) => {
    const signals = interpretedByTurn.get(event.turn_index) ?? {};
    const outcomePayload = event.payload?.outcome?.payload ?? {};

    return {
      index,
      turn: event.turn_index,
      label: `t${event.turn_index}`,
      reward: Number(event.payload?.reward ?? 0),
      comprehension_score: Number(signals.comprehension_score ?? 0),
      confusion_score: Number(signals.confusion_score ?? 0),
      checkpoint_correct:
        outcomePayload.checkpoint_correct == null ? null : Number(outcomePayload.checkpoint_correct),
      response_type: outcomePayload.response_type ?? "n/a",
    };
  });

  const responseCounts = new Map();
  series.forEach((entry) => {
    responseCounts.set(entry.response_type, (responseCounts.get(entry.response_type) ?? 0) + 1);
  });

  const responseBreakdown = [...responseCounts.entries()].map(([responseType, count]) => ({
    responseType,
    count,
  }));

  const log = series.map((entry) => ({
    ...entry,
    checkpointLabel:
      entry.checkpoint_correct == null
        ? "n/a"
        : entry.checkpoint_correct > 0
          ? "correct"
          : "incorrect",
  }));

  return { series, responseBreakdown, log };
}

export function buildBenchmarkSummary(sessionEvents) {
  const rewards = eventSeries(sessionEvents, "reward.computed").map((event) =>
    Number(event.payload?.reward ?? 0)
  );
  const actionScored = eventSeries(sessionEvents, "action.scored");
  const interpreted = eventSeries(sessionEvents, "outcome.interpreted");

  const totalReward = rewards.reduce((sum, value) => sum + value, 0);
  const averageReward = rewards.length ? totalReward / rewards.length : 0;
  const lastSignals = interpreted[interpreted.length - 1]?.payload?.signals ?? {};
  const policyType =
    actionScored[actionScored.length - 1]?.payload?.policy_info?.policy_type ?? "n/a";

  return [
    { label: "Policy", value: policyType },
    { label: "Events", value: String(sessionEvents.length) },
    { label: "Avg reward", value: formatNumber(averageReward) },
    {
      label: "Comprehension",
      value: lastSignals.comprehension_score == null ? "n/a" : formatNumber(lastSignals.comprehension_score),
    },
  ];
}

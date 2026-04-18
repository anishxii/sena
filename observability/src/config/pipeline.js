import { Activity, Bot, Brain, Sparkles, Waves } from "lucide-react";
import { formatNumber } from "../data";

export const pipelineNodes = [
  {
    id: "rawObservation",
    label: "Raw Observation",
    subtitle: "EEG, behavior, context",
    icon: Waves,
    status: "warning",
    x: 10,
    y: 50,
  },
  {
    id: "cognitiveState",
    label: "Cognitive State",
    subtitle: "Generalized latent representation",
    icon: Brain,
    status: "normal",
    x: 30,
    y: 50,
  },
  {
    id: "policyDecision",
    label: "Policy Decision",
    subtitle: "Action scoring and selection",
    icon: Sparkles,
    status: "critical",
    x: 50,
    y: 50,
  },
  {
    id: "applicationEffect",
    label: "Application Effect",
    subtitle: "Renderable intervention semantics",
    icon: Bot,
    status: "normal",
    x: 70,
    y: 50,
  },
  {
    id: "outcome",
    label: "Observed Outcome",
    subtitle: "Feedback and reward signals",
    icon: Activity,
    status: "normal",
    x: 90,
    y: 50,
  },
];

export const pipelineEdges = [
  ["rawObservation", "cognitiveState"],
  ["cognitiveState", "policyDecision"],
  ["policyDecision", "applicationEffect"],
  ["applicationEffect", "outcome"],
];

export function summarizeSnapshot(snapshot) {
  return {
    rawObservation: snapshot.rawObservation,
    cognitiveState: snapshot.cognitiveState,
    policyDecision: {
      selected_action: snapshot.selectedAction,
      policy_mode: snapshot.policyMode,
      scores: snapshot.actionScores,
      rationale: snapshot.rationale,
    },
    applicationEffect: snapshot.applicationEffect,
    outcome: snapshot.outcome,
  };
}

export function getNodeMetric(nodeId, snapshot) {
  switch (nodeId) {
    case "rawObservation":
      return `${formatNumber(snapshot.rawObservation?.eeg?.workload_estimate)} load`;
    case "cognitiveState":
      return `${Object.keys(snapshot.cognitiveState || {}).length} dimensions`;
    case "policyDecision":
      return snapshot.selectedAction;
    case "applicationEffect":
      return snapshot.applicationEffect?.effect_type || "n/a";
    case "outcome":
      return `reward ${formatNumber(snapshot.reward)}`;
    default:
      return "n/a";
  }
}

export function getNodeDetail(nodeId, snapshot) {
  switch (nodeId) {
    case "rawObservation":
      return `theta/alpha ${formatNumber(snapshot.rawObservation?.eeg?.theta_alpha_ratio)}`;
    case "cognitiveState":
      return `readiness ${formatNumber(snapshot.cognitiveState?.readiness_for_depth)}`;
    case "policyDecision":
      return `policy ${snapshot.policyMode}`;
    case "applicationEffect":
      return snapshot.applicationEffect?.summary?.split(".")[0] || "intervention mapped";
    case "outcome":
      return snapshot.outcome?.explicit_feedback?.response_type || "response logged";
    default:
      return "";
  }
}

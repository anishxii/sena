import React from "react";
import { Brain, Lightbulb, Sparkles, TrendingUp, UserRound } from "lucide-react";

const SIGNAL_COLORS = {
  cognitive_load: "#ff7a59",
  confidence: "#3b82f6",
  curiosity: "#8b5cf6",
};

const SIGNAL_LABELS = {
  cognitive_load: "load",
  confidence: "confidence",
  curiosity: "curiosity",
};

const ACTION_META = {
  deepen: {
    icon: Brain,
    className: "action-token action-token-deepen",
  },
  analogy: {
    icon: Lightbulb,
    className: "action-token action-token-analogy",
  },
};

const DEMO_THREADS = [
  {
    id: "learner_a",
    label: "Learner A",
    note: "Lower load, high confidence, ready for deeper abstraction.",
    summary: [
      { icon: Brain, label: "dominant state", value: "low load" },
      { icon: Sparkles, label: "selected action", value: "deepen" },
      { icon: TrendingUp, label: "reward", value: "+0.46" },
    ],
    timeline: [
      { position: 0.0, cognitive_load: 0.34, confidence: 0.70, curiosity: 0.58 },
      { position: 0.08, cognitive_load: 0.31, confidence: 0.72, curiosity: 0.60 },
      { position: 0.16, cognitive_load: 0.29, confidence: 0.74, curiosity: 0.62 },
      { position: 0.24, cognitive_load: 0.28, confidence: 0.76, curiosity: 0.64 },
      { position: 0.32, cognitive_load: 0.30, confidence: 0.75, curiosity: 0.66 },
      { position: 0.4, cognitive_load: 0.27, confidence: 0.78, curiosity: 0.67 },
      { position: 0.48, cognitive_load: 0.26, confidence: 0.79, curiosity: 0.69 },
      { position: 0.56, cognitive_load: 0.25, confidence: 0.81, curiosity: 0.70 },
      { position: 0.64, cognitive_load: 0.24, confidence: 0.82, curiosity: 0.71 },
      { position: 0.72, cognitive_load: 0.23, confidence: 0.83, curiosity: 0.72 },
      { position: 0.8, cognitive_load: 0.22, confidence: 0.84, curiosity: 0.73 },
      { position: 0.88, cognitive_load: 0.23, confidence: 0.83, curiosity: 0.74 },
      { position: 0.96, cognitive_load: 0.21, confidence: 0.85, curiosity: 0.75 },
    ],
    messages: [
      {
        role: "user",
        text: "Let's talk about gradient descent.",
      },
      {
        role: "tutor",
        text:
          "Gradient descent works because the gradient is a local directional derivative: it tells us how sharply loss rises around the current parameter setting, so stepping against it gives the steepest immediate decrease.",
        metaLeft: "action",
        metaValue: "deepen",
      },
      {
        role: "user",
        text: "Yes, this makes sense. So the gradient is really the mechanism that turns loss into motion.",
        metaLeft: "reward",
        metaValue: "+0.46",
      },
      {
        role: "tutor",
        text:
          "Exactly. Once that mechanism is clear, learning rate becomes a stability question: not which way to move, but how far to trust the local slope before the landscape changes.",
      },
      {
        role: "user",
        text: "That helps. Now I can connect gradient, step size, and overshooting in one picture.",
      },
    ],
  },
  {
    id: "learner_b",
    label: "Learner B",
    note: "Higher load, weaker confidence, same opening behavior but different hidden state.",
    summary: [
      { icon: Brain, label: "dominant state", value: "high load" },
      { icon: Sparkles, label: "selected action", value: "analogy" },
      { icon: TrendingUp, label: "reward", value: "+0.09" },
    ],
    timeline: [
      { position: 0.0, cognitive_load: 0.76, confidence: 0.43, curiosity: 0.56 },
      { position: 0.08, cognitive_load: 0.78, confidence: 0.41, curiosity: 0.55 },
      { position: 0.16, cognitive_load: 0.75, confidence: 0.40, curiosity: 0.54 },
      { position: 0.24, cognitive_load: 0.73, confidence: 0.39, curiosity: 0.53 },
      { position: 0.32, cognitive_load: 0.74, confidence: 0.38, curiosity: 0.52 },
      { position: 0.4, cognitive_load: 0.71, confidence: 0.40, curiosity: 0.54 },
      { position: 0.48, cognitive_load: 0.69, confidence: 0.41, curiosity: 0.55 },
      { position: 0.56, cognitive_load: 0.68, confidence: 0.42, curiosity: 0.56 },
      { position: 0.64, cognitive_load: 0.66, confidence: 0.43, curiosity: 0.57 },
      { position: 0.72, cognitive_load: 0.67, confidence: 0.44, curiosity: 0.58 },
      { position: 0.8, cognitive_load: 0.65, confidence: 0.45, curiosity: 0.59 },
      { position: 0.88, cognitive_load: 0.63, confidence: 0.46, curiosity: 0.60 },
      { position: 0.96, cognitive_load: 0.62, confidence: 0.47, curiosity: 0.61 },
    ],
    messages: [
      {
        role: "user",
        text: "Let's talk about gradient descent.",
      },
      {
        role: "tutor",
        text:
          "Think of gradient descent like hiking down a foggy mountain. You cannot see the whole valley, but you can still feel which direction slopes downward right where you are standing.",
        metaLeft: "action",
        metaValue: "analogy",
      },
      {
        role: "user",
        text: "This part confuses me a bit. I get the mountain image, but I still do not see how that becomes an actual parameter update.",
        metaLeft: "reward",
        metaValue: "+0.09",
      },
      {
        role: "tutor",
        text:
          "Then keep the analogy, but pin it to the equation: the gradient gives the downhill direction, and the learning rate controls how large the downhill step is on each move.",
      },
      {
        role: "user",
        text: "Okay, that is a little clearer. I follow the direction part better than before.",
      },
    ],
  },
];

function MetaPill({ icon: Icon, label, value }) {
  return (
    <div className="chat-meta-pill">
      <Icon className="icon-14" />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function buildSignalPath(timeline, signalKey, width, height, paddingX, paddingY) {
  if (!timeline.length) return "";
  const points = timeline.map((snapshot) => ({
    x: paddingX + snapshot[signalKey] * (width - paddingX * 2),
    y: paddingY + snapshot.position * (height - paddingY * 2),
  }));

  if (points.length === 1) {
    return `M ${points[0].x} ${points[0].y}`;
  }

  let path = `M ${points[0].x} ${points[0].y}`;
  for (let index = 0; index < points.length - 1; index += 1) {
    const current = points[index];
    const next = points[index + 1];
    const controlY = current.y + (next.y - current.y) / 2;
    path += ` C ${current.x} ${controlY}, ${next.x} ${controlY}, ${next.x} ${next.y}`;
  }
  return path;
}

function SignalTimeline({ timeline }) {
  const width = 122;
  const height = 316;
  const paddingX = 18;
  const paddingY = 18;

  return (
    <div className="signal-timeline">
      <div className="signal-timeline-label">cognitive rail</div>
      <div className="signal-graph-shell">
        <svg viewBox={`0 0 ${width} ${height}`} className="signal-graph">
          {[0.1, 0.3, 0.5, 0.7, 0.9].map((position) => {
            const y = paddingY + position * (height - paddingY * 2);
            return (
              <line
                key={`grid-${position}`}
                x1={paddingX}
                x2={width - paddingX}
                y1={y}
                y2={y}
                className="signal-grid-line"
              />
            );
          })}

          {Object.entries(SIGNAL_COLORS).map(([signalKey, color]) => (
            <path
              key={signalKey}
              d={buildSignalPath(timeline, signalKey, width, height, paddingX, paddingY)}
              fill="none"
              stroke={color}
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          ))}

          {timeline.flatMap((snapshot, index) =>
            Object.entries(SIGNAL_COLORS).map(([signalKey, color]) => {
              const x = paddingX + snapshot[signalKey] * (width - paddingX * 2);
              const y = paddingY + snapshot.position * (height - paddingY * 2);
              return (
                <circle
                  key={`${signalKey}-${index}`}
                  cx={x}
                  cy={y}
                  r="2.6"
                  fill={color}
                  stroke="#ffffff"
                  strokeWidth="1.1"
                />
              );
            })
          )}
        </svg>
      </div>
      <div className="signal-legend">
        {Object.entries(SIGNAL_COLORS).map(([signalKey, color]) => (
          <span key={signalKey}>
            <i style={{ background: color }} />
            {SIGNAL_LABELS[signalKey]}
          </span>
        ))}
      </div>
    </div>
  );
}

function MessageBubble({ message }) {
  const isUser = message.role === "user";
  const actionMeta = !isUser && message.metaLeft === "action" ? ACTION_META[message.metaValue] : null;
  const ActionIcon = actionMeta?.icon;

  return (
    <div className={isUser ? "chat-row chat-row-user" : "chat-row chat-row-tutor"}>
      <div className={isUser ? "chat-bubble chat-bubble-user" : "chat-bubble chat-bubble-tutor"}>
        <div className="chat-bubble-role">
          {isUser ? <UserRound className="icon-14" /> : <Sparkles className="icon-14" />}
          <span>{isUser ? "User" : "Tutor"}</span>
        </div>
        <p>{message.text}</p>
        {message.metaLeft ? (
          <div className="chat-bubble-meta">
            <span>{message.metaLeft}</span>
            {actionMeta ? (
              <strong className={actionMeta.className}>
                <ActionIcon className="icon-14" />
                {message.metaValue}
              </strong>
            ) : (
              <strong>{message.metaValue}</strong>
            )}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function ChatPane({ thread }) {
  return (
    <section className="chat-pane">
      <header className="chat-pane-header">
        <div>
          <p className="chat-pane-kicker">same opening prompt</p>
          <h2 className="chat-pane-title">{thread.label}</h2>
          <p className="chat-pane-note">{thread.note}</p>
        </div>
        <div className="chat-pane-summary">
          {thread.summary.map((item) => (
            <MetaPill key={item.label} icon={item.icon} label={item.label} value={item.value} />
          ))}
        </div>
      </header>

      <div className="chat-pane-body">
        <div className="chat-conversation">
          {thread.messages.map((message, index) => (
            <MessageBubble key={`${thread.id}-${index}`} message={message} />
          ))}
        </div>
        <SignalTimeline timeline={thread.timeline} />
      </div>
    </section>
  );
}

export default function ChatComparisonApp() {
  return (
    <main className="chat-shell">
      <header className="chat-hero">
        <p className="chat-hero-kicker">Static comparison artifact</p>
        <h1 className="chat-hero-title">Sena</h1>
      </header>

      <section className="chat-columns">
        {DEMO_THREADS.map((thread) => (
          <ChatPane key={thread.id} thread={thread} />
        ))}
      </section>
    </main>
  );
}

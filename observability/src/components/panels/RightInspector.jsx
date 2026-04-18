import React, { useMemo, useState } from "react";
import { ChevronDown } from "lucide-react";
import { pipelineNodes } from "../../config/pipeline";

function prettyJson(value) {
  return JSON.stringify(value, null, 2);
}

function findNode(nodeId) {
  return pipelineNodes.find((node) => node.id === nodeId);
}

function MetaRow({ label, value }) {
  return (
    <div className="inspector-meta-row">
      <span>{label}</span>
      <span>{value ?? "n/a"}</span>
    </div>
  );
}

function CollapsibleSection({ label, title, defaultOpen = true, children }) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <section className="inspector-card inspector-card-grow">
      <button type="button" className="inspector-section-toggle" onClick={() => setOpen((current) => !current)}>
        <div>
          <p className="inspector-card-kicker">{label}</p>
          <h2 className="inspector-card-title">{title}</h2>
        </div>
        <ChevronDown className={`icon-16 inspector-chevron ${open ? "is-open" : ""}`} />
      </button>
      {open ? children : null}
    </section>
  );
}

export function RightInspector({ canvasView, selectedNodeId, activeEventIndex, totalEvents }) {
  const selectedNode = findNode(selectedNodeId);
  const selectedNodeView = canvasView.nodeViews[selectedNodeId];
  const selectedNodeEvent = canvasView.nodePayloads[selectedNodeId];
  const rawPayload = useMemo(() => prettyJson(selectedNodeEvent?.payload ?? {}), [selectedNodeEvent]);
  const rawSummary = useMemo(() => prettyJson(selectedNodeEvent?.summary ?? {}), [selectedNodeEvent]);

  return (
    <aside className="stream-inspector">
      <section className="inspector-card">
        <p className="inspector-card-kicker">Current event</p>
        <h2 className="inspector-card-title">{canvasView.currentEventType}</h2>
        <div className="inspector-meta-list">
          <MetaRow label="turn" value={canvasView.turnIndex} />
          <MetaRow label="event" value={`${activeEventIndex + 1} / ${totalEvents}`} />
          <MetaRow label="session" value={canvasView.sessionLabel} />
          <MetaRow label="context" value={canvasView.contextLabel} />
          <MetaRow label="stage" value={canvasView.currentEvent?.stage} />
          <MetaRow label="source" value={canvasView.currentEvent?.source} />
        </div>
      </section>

      <section className="inspector-card">
        <p className="inspector-card-kicker">Selected node</p>
        <h2 className="inspector-card-title">{selectedNode?.label ?? selectedNodeId}</h2>
        <div className="inspector-node-summary">
          <div className="inspector-summary-pill">
            <span>metric</span>
            <strong>{selectedNodeView?.metric ?? "n/a"}</strong>
          </div>
          <div className="inspector-summary-detail">{selectedNodeView?.detail ?? "n/a"}</div>
          <div className="inspector-summary-subtitle">{selectedNode?.subtitle ?? ""}</div>
        </div>
      </section>

      <CollapsibleSection label="Summary" title={selectedNodeEvent?.event_type ?? "n/a"} defaultOpen={true}>
        <pre className="inspector-json inspector-json-compact">{rawSummary}</pre>
      </CollapsibleSection>

      <CollapsibleSection label="Payload" title="Raw structured payload" defaultOpen={false}>
        <pre className="inspector-json">{rawPayload}</pre>
      </CollapsibleSection>
    </aside>
  );
}

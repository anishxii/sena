import React from "react";
import { formatNumber } from "../../data";

export function RightPanel({ snapshot, selectedNode }) {
  return (
    <aside className="inspector">
      <div className="inspector-header">
        <div className="inspector-meta">selected node</div>
        <div className="inspector-title">{selectedNode}</div>
        <p className="inspector-copy">
          This rail will become the deep inspector. For now, it keeps the canvas readable while still showing branch context.
        </p>
      </div>

      <div className="inspector-scroll">
        <section className="info-card">
          <p className="info-kicker">Current branch</p>
          <h3 className="info-title">{snapshot.userId} · {snapshot.policyMode}</h3>
          <div className="info-rows">
            <div className="info-row"><span>Reward</span><span>{formatNumber(snapshot.reward)}</span></div>
            <div className="info-row"><span>Action</span><span>{snapshot.selectedAction}</span></div>
            <div className="info-row"><span>Context</span><span>{snapshot.contextSummary}</span></div>
          </div>
        </section>
      </div>
    </aside>
  );
}

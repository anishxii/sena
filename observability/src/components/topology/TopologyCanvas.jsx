import React from "react";
import { Pause, Play } from "lucide-react";
import { ReactFlow, BaseEdge, getStraightPath } from "@xyflow/react";
import { NodeOverlayPanel } from "../panels/NodeOverlayPanel";
import { pipelineEdges, pipelineNodes } from "../../config/pipeline";
import { FlowTopologyNode } from "./TopologyNode";

function FlowSignalEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  style,
  markerEnd,
  data,
}) {
  const [edgePath] = getStraightPath({ sourceX, sourceY, targetX, targetY });
  const isDimmed = data?.dimmed;
  const isHighlighted = data?.highlighted;

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd}
        style={{
          ...style,
          stroke: isHighlighted ? "rgba(196,181,253,0.84)" : "rgba(162,173,255,0.22)",
          strokeWidth: isHighlighted ? 1.6 : 1.15,
          opacity: isDimmed ? 0.18 : 1,
        }}
      />
      <circle
        r="3.5"
        fill={isHighlighted ? "rgba(255,255,255,0.95)" : "rgba(196,181,253,0.72)"}
        opacity={isDimmed ? 0.14 : 0.92}
      >
        <animateMotion dur="1.8s" repeatCount="indefinite" path={edgePath} />
      </circle>
    </>
  );
}

const nodeTypes = { topologyNode: FlowTopologyNode };
const edgeTypes = { signal: FlowSignalEdge };

export function TopologyCanvas({
  canvasView,
  sessionEvents,
  replayEvents,
  activeEventIndex,
  onChangeEventIndex,
  isPlaying,
  onTogglePlay,
  selectedNodeId,
  onSelectNode,
  openPanelNodeId,
  onClosePanel,
  benchmarks,
}) {
  const highlightedEdges = new Set(
    pipelineEdges
      .filter(([left, right]) => left === selectedNodeId || right === selectedNodeId)
      .map(([left, right]) => `${left}-${right}`)
  );

  const nodes = pipelineNodes.map((node, index) => {
    const isSelected = node.id === selectedNodeId;
    const isConnected =
      pipelineEdges.some(([left, right]) => left === selectedNodeId && right === node.id) ||
      pipelineEdges.some(([left, right]) => right === selectedNodeId && left === node.id) ||
      isSelected;

    return {
      id: node.id,
      type: "topologyNode",
      position: { x: 48 + index * 268, y: 188 },
      draggable: false,
      selectable: true,
      connectable: false,
      data: {
        node,
        metric: canvasView.nodeViews[node.id]?.metric ?? "n/a",
        detail: canvasView.nodeViews[node.id]?.detail ?? "",
        dimmed: selectedNodeId ? !isConnected : false,
        onSelect: onSelectNode,
      },
    };
  });

  const edges = pipelineEdges.map(([source, target], index) => {
    const edgeKey = `${source}-${target}`;
    const highlighted = highlightedEdges.has(edgeKey);
    const dimmed = !highlighted && Boolean(selectedNodeId);

    return {
      id: edgeKey,
      source,
      target,
      type: "signal",
      animated: false,
      selectable: false,
      style: { zIndex: index },
      data: { highlighted, dimmed },
    };
  });

  return (
    <section className="topology-stage">
      <div className="topology-stage-header">
        <div>
          <p className="canvas-kicker">Session replay</p>
          <h2 className="canvas-title">{canvasView.sessionLabel}</h2>
          <p className="canvas-copy">
            Turn {canvasView.turnIndex} · {canvasView.currentEventType} · {canvasView.contextLabel}
          </p>
        </div>
        <div className="canvas-header-meta">
          <div className="canvas-benchmark-strip">
            {(benchmarks ?? []).map((benchmark) => (
              <div key={benchmark.label} className="canvas-benchmark-card">
                <span>{benchmark.label}</span>
                <strong>{benchmark.value}</strong>
              </div>
            ))}
          </div>
          <div className="canvas-event-pill">
            event {activeEventIndex + 1}/{sessionEvents.length}
          </div>
          <button type="button" className="canvas-play-button" aria-label="Play simulation replay" onClick={onTogglePlay}>
            {isPlaying ? <Pause className="icon-16" /> : <Play className="icon-16" />}
          </button>
        </div>
      </div>

      <div className="topology-frame">
        <div className="canvas-grid" />
        <div className="canvas-glow canvas-glow-a" />
        <div className="canvas-glow canvas-glow-b" />
        <div className="flow-stage" aria-label="Cognitive middleware pipeline">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            fitView
            fitViewOptions={{ padding: 0.12, minZoom: 1, maxZoom: 1 }}
            nodesDraggable={false}
            nodesConnectable={false}
            elementsSelectable={false}
            zoomOnScroll={false}
            zoomOnPinch={false}
            panOnDrag={false}
            panOnScroll={false}
            zoomOnDoubleClick={false}
            preventScrolling={false}
            proOptions={{ hideAttribution: true }}
          />
        </div>
        {openPanelNodeId ? (
          <NodeOverlayPanel
            nodeId={openPanelNodeId}
            replayEvents={replayEvents}
            canvasView={canvasView}
            onClose={onClosePanel}
          />
        ) : null}
      </div>

      <div className="replay-scrubber replay-scrubber-floating">
        <div className="scrubber-labels">
          <span>Replay index</span>
          <span>
            {activeEventIndex + 1} / {sessionEvents.length}
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={Math.max(0, sessionEvents.length - 1)}
          value={activeEventIndex}
          onChange={(event) => onChangeEventIndex(Number(event.target.value))}
          className="scrubber-range"
        />
      </div>
    </section>
  );
}

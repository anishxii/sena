import React, { useEffect, useMemo, useState } from "react";
import { TopologyCanvas } from "./components/topology/TopologyCanvas";
import { getReplayWindow, getSessionEvents, listSessions, loadSampleStreamEvents } from "./replay/sampleStream";
import { buildBenchmarkSummary, buildCanvasView } from "./replay/viewModel";

export default function App() {
  const events = useMemo(() => loadSampleStreamEvents(), []);
  const sessionIds = useMemo(() => listSessions(events), [events]);
  const [selectedSessionId] = useState(sessionIds[0] ?? null);
  const [selectedNodeId, setSelectedNodeId] = useState("policyDecision");
  const [openPanelNodeId, setOpenPanelNodeId] = useState(null);
  const [activeEventIndex, setActiveEventIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const sessionEvents = useMemo(
    () => (selectedSessionId ? getSessionEvents(events, selectedSessionId) : []),
    [events, selectedSessionId]
  );
  const replayWindow = useMemo(
    () => getReplayWindow(sessionEvents, activeEventIndex),
    [sessionEvents, activeEventIndex]
  );
  const canvasView = useMemo(
    () => buildCanvasView(replayWindow.visibleEvents),
    [replayWindow.visibleEvents]
  );
  const benchmarks = useMemo(
    () => buildBenchmarkSummary(replayWindow.visibleEvents),
    [replayWindow.visibleEvents]
  );

  useEffect(() => {
    if (!isPlaying || sessionEvents.length <= 1) return undefined;
    const interval = window.setInterval(() => {
      setActiveEventIndex((current) => {
        if (current >= sessionEvents.length - 1) {
          setIsPlaying(false);
          window.clearInterval(interval);
          return current;
        }
        return current + 1;
      });
    }, 900);
    return () => window.clearInterval(interval);
  }, [isPlaying, sessionEvents.length]);

  return (
    <div className="dashboard-root">
      <div className="canvas-only-shell">
        <main className="canvas-only-main">
          <div className="canvas-only-header">
            <p className="canvas-only-kicker">Observability surface</p>
            <h1 className="canvas-only-title">Sena</h1>
          </div>
          <TopologyCanvas
            canvasView={canvasView}
            sessionEvents={sessionEvents}
            replayEvents={replayWindow.visibleEvents}
            activeEventIndex={replayWindow.clampedIndex}
            onChangeEventIndex={setActiveEventIndex}
            isPlaying={isPlaying}
            onTogglePlay={() => {
              if (replayWindow.clampedIndex >= sessionEvents.length - 1) {
                setActiveEventIndex(0);
              }
              setIsPlaying((current) => !current);
            }}
            selectedNodeId={selectedNodeId}
            onSelectNode={(nodeId) => {
              setSelectedNodeId(nodeId);
              if (
                nodeId === "rawObservation" ||
                nodeId === "cognitiveState" ||
                nodeId === "policyDecision" ||
                nodeId === "outcome"
              ) {
                setOpenPanelNodeId(nodeId);
              }
            }}
            openPanelNodeId={openPanelNodeId}
            onClosePanel={() => setOpenPanelNodeId(null)}
            benchmarks={benchmarks}
          />
        </main>
      </div>
    </div>
  );
}

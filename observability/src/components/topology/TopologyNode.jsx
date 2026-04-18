import React from "react";
import { motion } from "framer-motion";
import { Handle, Position } from "@xyflow/react";
import { cn } from "../../lib/cn";

function statusClasses(status) {
  switch (status) {
    case "critical":
      return "node-critical";
    case "warning":
      return "node-warning";
    case "normal":
      return "node-normal";
    default:
      return "node-muted";
  }
}

export function TopologyNode({ node, metric, detail, selected, dimmed, onSelect }) {
  const Icon = node.icon;

  return (
    <motion.button
      type="button"
      className={cn("topology-node", statusClasses(node.status), selected && "is-selected")}
      onClick={() => onSelect(node.id)}
      initial={false}
      animate={{ scale: selected ? 1.02 : 1, opacity: dimmed ? 0.38 : 1 }}
      transition={{ type: "spring", stiffness: 260, damping: 24 }}
    >
      <div className="topology-node-top">
        <div className="topology-node-icon">
          <Icon className="icon-16" />
        </div>
        <div className="topology-node-metric">{metric}</div>
      </div>
      <div className="topology-node-title">{node.label}</div>
      <div className="topology-node-subtitle">{node.subtitle}</div>
      <div className="topology-node-detail">{detail}</div>
    </motion.button>
  );
}

export function FlowTopologyNode({ id, data, selected }) {
  const Icon = data.node.icon;

  return (
    <motion.button
      type="button"
      className={cn("topology-node", statusClasses(data.node.status), selected && "is-selected")}
      onClick={() => data.onSelect(id)}
      initial={false}
      animate={{ scale: selected ? 1.02 : 1, opacity: data.dimmed ? 0.38 : 1 }}
      transition={{ type: "spring", stiffness: 260, damping: 24 }}
    >
      <Handle type="target" position={Position.Left} className="rf-handle rf-handle-left" />
      <div className="topology-node-top">
        <div className="topology-node-icon">
          <Icon className="icon-16" />
        </div>
        <div className="topology-node-metric">{data.metric}</div>
      </div>
      <div className="topology-node-title">{data.node.label}</div>
      <div className="topology-node-subtitle">{data.node.subtitle}</div>
      <div className="topology-node-detail">{data.detail}</div>
      <Handle type="source" position={Position.Right} className="rf-handle rf-handle-right" />
    </motion.button>
  );
}

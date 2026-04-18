import {
  Activity,
  BarChart3,
  Bot,
  FileSearch,
  Globe,
  Home,
  Network,
  Radar,
  ShieldAlert,
  Waves,
} from "lucide-react";

export const navGroups = [
  {
    label: "Overview",
    items: [
      { icon: Home, label: "Home" },
      { icon: Bot, label: "Applications" },
      { icon: BarChart3, label: "Dashboards" },
      { icon: Network, label: "Integrations" },
    ],
  },
  {
    label: "Monitoring",
    items: [
      { icon: Radar, label: "Cognitive Map", active: true },
      { icon: Globe, label: "Experiments" },
      { icon: Activity, label: "Streams" },
      { icon: ShieldAlert, label: "Alerts" },
    ],
  },
  {
    label: "Telemetry",
    items: [
      { icon: FileSearch, label: "Query State" },
      { icon: BarChart3, label: "Metrics" },
      { icon: Waves, label: "Signals" },
    ],
  },
];

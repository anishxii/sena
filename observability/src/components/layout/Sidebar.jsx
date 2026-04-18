import React from "react";
import { Bell, Brain } from "lucide-react";
import { navGroups } from "../../config/navigation";
import { cn } from "../../lib/cn";

export function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="sidebar-logo">
          <Brain className="icon-16" />
        </div>
        <div className="sidebar-brand-text">middleware</div>
      </div>

      <div className="sidebar-groups">
        {navGroups.map((group) => (
          <div key={group.label} className="sidebar-group">
            <div className="sidebar-group-label">{group.label}</div>
            <div className="sidebar-group-items">
              {group.items.map((item) => {
                const Icon = item.icon;
                return (
                  <button key={item.label} className={cn("sidebar-item", item.active && "is-active")} type="button">
                    <Icon className="icon-16" />
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      <div className="sidebar-footer">
        <button className="sidebar-item" type="button">
          <Bell className="icon-16" />
          <span>Notifications</span>
        </button>
      </div>
    </aside>
  );
}

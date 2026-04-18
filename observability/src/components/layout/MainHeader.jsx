import React from "react";
import { ChevronRight, Clock3, Command, Search, Settings2 } from "lucide-react";
import { Button } from "../ui/Button";

export function MainHeader() {
  return (
    <>
      <div className="main-header">
        <div>
          <div className="breadcrumb">
            <span>Built-in views</span>
            <ChevronRight className="icon-12" />
            <span>All applications</span>
          </div>
          <h1 className="page-title">Cognitive middleware topology</h1>
          <p className="page-subtitle">Trace how observation becomes state, action, and outcome through one shared control surface.</p>
        </div>

        <div className="header-actions">
          <Button>
            <Command className="icon-16" />
            Search
          </Button>
          <Button>
            <Settings2 className="icon-16" />
            Default view
          </Button>
          <Button>
            <Clock3 className="icon-16" />
            Demo replay
          </Button>
        </div>
      </div>

      <div className="toolbar-row">
        <div className="search-shell">
          <Search className="search-icon" />
          <input className="search-input" placeholder="Search nodes, users, or traces..." />
        </div>
        <Button>Map settings</Button>
      </div>
    </>
  );
}

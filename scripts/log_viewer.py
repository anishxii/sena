from __future__ import annotations

import json
from html import escape
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.end_to_end_demo import ARTIFACTS_DIR, TURN_LOG_PATH


HTML_PATH = ARTIFACTS_DIR / "turn_log_viewer.html"


def _pretty_json(value) -> str:
    return escape(json.dumps(value, indent=2))


def build_html(turn_logs: list[dict]) -> str:
    cards = []
    for index, turn in enumerate(turn_logs, start=1):
        action_scores = turn["action_scores"]["scores"]
        scores_html = "".join(
            f"<li><strong>{escape(action_id)}</strong>: {score:.3f}</li>"
            for action_id, score in action_scores.items()
        )
        cards.append(
            f"""
            <section class="card">
              <h2>Turn {index}</h2>
              <div class="grid">
                <div>
                  <h3>System 3: Raw Observation</h3>
                  <pre>{_pretty_json(turn["raw_observation"])}</pre>
                </div>
                <div>
                  <h3>System 2: State</h3>
                  <pre>{_pretty_json(turn["state"])}</pre>
                </div>
                <div>
                  <h3>System 1: Action Scores</h3>
                  <p><strong>Selected:</strong> {escape(turn["action"]["action_id"])}</p>
                  <p><strong>Policy:</strong> {escape(turn["action_scores"]["policy_info"]["policy_type"])}</p>
                  <p><strong>Exploration:</strong> {turn["action_scores"]["policy_info"]["exploration"]}</p>
                  <ul>{scores_html}</ul>
                </div>
                <div>
                  <h3>System 2: Interaction Effect</h3>
                  <pre>{_pretty_json(turn["interaction_effect"])}</pre>
                </div>
                <div>
                  <h3>System 3: Outcome</h3>
                  <pre>{_pretty_json(turn["outcome"])}</pre>
                </div>
                <div>
                  <h3>System 2 -> System 1: Reward Event</h3>
                  <pre>{_pretty_json(turn["reward_event"])}</pre>
                </div>
              </div>
            </section>
            """
        )

    return f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <title>Emotiv Learn Turn Log Viewer</title>
        <style>
          body {{
            font-family: ui-sans-serif, system-ui, sans-serif;
            background: #f5f7fb;
            color: #162032;
            margin: 0;
            padding: 24px;
          }}
          h1 {{
            margin-top: 0;
          }}
          .card {{
            background: white;
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 12px 32px rgba(20, 31, 58, 0.08);
          }}
          .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 16px;
          }}
          pre {{
            background: #0f172a;
            color: #e2e8f0;
            padding: 12px;
            border-radius: 12px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-word;
          }}
          ul {{
            padding-left: 20px;
          }}
        </style>
      </head>
      <body>
        <h1>Emotiv Learn Turn Log Viewer</h1>
        <p>This viewer shows the full smoke-style handoff across System 3, System 2, and System 1 for each turn.</p>
        {''.join(cards)}
      </body>
    </html>
    """


def main() -> None:
    if not TURN_LOG_PATH.exists():
        raise SystemExit(
            f"Missing turn log file at {TURN_LOG_PATH}. Run `python3 scripts/end_to_end_demo.py` first."
        )

    turn_logs = json.loads(TURN_LOG_PATH.read_text(encoding="utf-8"))
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    HTML_PATH.write_text(build_html(turn_logs), encoding="utf-8")
    print(f"Wrote viewer to {HTML_PATH}")


if __name__ == "__main__":
    main()

"""
Flask Web Interface for Training Simulation
============================================
ChatGPT-like interface for running and monitoring multiple training simulations.

Usage:
    python web_interface.py

Then open http://localhost:5000 in your browser
"""

import json
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from simulated_user import SimulatedHumanAgent
from train_simulation import (
    generate_lesson_plan,
    generate_content_step,
    CONTENT_ACTIONS,
)

load_dotenv()

app = Flask(__name__)

# Store active simulations in memory
# Format: {session_id: {metadata, steps, status, paused, etc.}}
simulations: Dict[str, Dict] = {}
simulations_lock = threading.Lock()

# Pause events for each simulation
pause_events: Dict[str, threading.Event] = {}


# ── Helper Functions ───────────────────────────────────────────────────────────

def get_available_subjects(stew_dir: str = "../stew_dataset") -> List[str]:
    """Get list of available STEW subjects."""
    subjects = []
    for fname in sorted(os.listdir(stew_dir)):
        if fname.endswith("_hi.txt"):
            subj_id = fname.replace("_hi.txt", "")
            subjects.append(subj_id)
    return subjects


def run_simulation_thread(
    session_id: str,
    topic: str,
    num_steps: int,
    subject_id: str | None,
    stew_dir: str,
    seed: int,
):
    """
    Run a simulation in a background thread.
    Updates the simulations dict as it progresses.
    """
    try:
        rng = np.random.default_rng(seed)

        # Create pause event for this simulation
        pause_event = threading.Event()
        pause_event.set()  # Start unpaused
        pause_events[session_id] = pause_event

        # Update status
        with simulations_lock:
            simulations[session_id]["status"] = "initializing"
            simulations[session_id]["progress"] = 0
            simulations[session_id]["paused"] = False

        # Initialize simulated user
        agent = SimulatedHumanAgent(
            stew_dir=stew_dir,
            subject_id=subject_id,
            seed=seed,
            loop_epochs=True,
        )

        with simulations_lock:
            simulations[session_id]["subject_id"] = agent.subject_id
            simulations[session_id]["subject_rating"] = float(agent.rating)
            simulations[session_id]["status"] = "generating_lesson_plan"

        # Generate lesson plan
        lesson_plan = generate_lesson_plan(topic, num_steps)

        with simulations_lock:
            simulations[session_id]["lesson_plan"] = lesson_plan
            simulations[session_id]["status"] = "running"

        # Run simulation loop
        previous_content = ""
        current_action = "maintain"

        for i, step_info in enumerate(lesson_plan):
            step_num = i + 1

            # Check if paused (wait until resumed)
            pause_event.wait()

            # Generate content
            content = generate_content_step(
                topic=topic,
                step_info=step_info,
                step_number=step_num,
                total_steps=num_steps,
                previous_content=previous_content,
                optimization_action=current_action,
            )

            # Simulated user processes content
            user_response = agent.step(content)

            # RL action (random for now)
            next_action = rng.choice(CONTENT_ACTIONS)

            # Store step data
            step_data = {
                "step_number": step_num,
                "step_title": step_info["title"],
                "content": content,
                "content_length": len(content),
                "optimization_action_applied": current_action,
                "eeg_features": user_response["eeg_features"].tolist(),
                "behavioral_cues": user_response["behavioral_cues"],
                "cognitive_load": user_response["cognitive_load"],
                "epochs_consumed": user_response["epochs_consumed"],
                "user_action": user_response["next_action"],
                "user_prompt": user_response["next_prompt"],
                "rl_action": next_action,
                "timestamp": datetime.now().isoformat(),
            }

            with simulations_lock:
                simulations[session_id]["steps"].append(step_data)
                simulations[session_id]["progress"] = int((step_num / num_steps) * 100)

            # Update for next iteration
            previous_content = content
            current_action = next_action

        # Mark as complete
        with simulations_lock:
            simulations[session_id]["status"] = "completed"
            simulations[session_id]["progress"] = 100
            simulations[session_id]["completed_at"] = datetime.now().isoformat()

    except Exception as e:
        with simulations_lock:
            simulations[session_id]["status"] = "error"
            simulations[session_id]["error"] = str(e)
    finally:
        # Clean up pause event
        if session_id in pause_events:
            del pause_events[session_id]


# ── Flask Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Main page with simulation interface."""
    return render_template("index.html")


@app.route("/api/subjects")
def api_subjects():
    """Get list of available STEW subjects."""
    try:
        subjects = get_available_subjects()
        return jsonify({"subjects": subjects})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/simulations", methods=["GET"])
def api_list_simulations():
    """List all active simulations."""
    with simulations_lock:
        sim_list = []
        for sid, sim in simulations.items():
            sim_list.append({
                "session_id": sid,
                "topic": sim["topic"],
                "status": sim["status"],
                "progress": sim["progress"],
                "subject_id": sim.get("subject_id"),
                "num_steps": sim["num_steps"],
                "current_step": len(sim["steps"]),
                "created_at": sim["created_at"],
            })
        return jsonify({"simulations": sim_list})


@app.route("/api/simulations/<session_id>", methods=["GET"])
def api_get_simulation(session_id):
    """Get detailed info for a specific simulation."""
    with simulations_lock:
        if session_id not in simulations:
            return jsonify({"error": "Simulation not found"}), 404
        return jsonify(simulations[session_id])


@app.route("/api/simulations/<session_id>/steps", methods=["GET"])
def api_get_steps(session_id):
    """Get all steps for a specific simulation."""
    with simulations_lock:
        if session_id not in simulations:
            return jsonify({"error": "Simulation not found"}), 404
        return jsonify({
            "steps": simulations[session_id]["steps"],
            "status": simulations[session_id]["status"],
            "progress": simulations[session_id]["progress"],
        })


@app.route("/api/simulations", methods=["POST"])
def api_start_simulation():
    """Start a new simulation."""
    data = request.json

    # Validate input
    topic = data.get("topic")
    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    num_steps = data.get("num_steps", 10)
    subject_id = data.get("subject_id")  # None for random
    stew_dir = data.get("stew_dir", "../stew_dataset")
    seed = data.get("seed", int(time.time()))

    # Create new simulation
    session_id = str(uuid.uuid4())[:8]

    simulation = {
        "session_id": session_id,
        "topic": topic,
        "num_steps": num_steps,
        "subject_id": subject_id,
        "stew_dir": stew_dir,
        "seed": seed,
        "status": "starting",
        "progress": 0,
        "paused": False,
        "steps": [],
        "created_at": datetime.now().isoformat(),
    }

    with simulations_lock:
        simulations[session_id] = simulation

    # Start simulation in background thread
    thread = threading.Thread(
        target=run_simulation_thread,
        args=(session_id, topic, num_steps, subject_id, stew_dir, seed),
        daemon=True,
    )
    thread.start()

    return jsonify({
        "session_id": session_id,
        "message": "Simulation started",
    })


@app.route("/api/simulations/<session_id>/pause", methods=["POST"])
def api_pause_simulation(session_id):
    """Pause or resume a simulation."""
    data = request.json
    action = data.get("action")  # "pause" or "resume"

    with simulations_lock:
        if session_id not in simulations:
            return jsonify({"error": "Simulation not found"}), 404

        if session_id not in pause_events:
            return jsonify({"error": "Cannot pause completed simulation"}), 400

        if action == "pause":
            pause_events[session_id].clear()  # Pause the simulation
            simulations[session_id]["status"] = "paused"
            simulations[session_id]["paused"] = True
        elif action == "resume":
            pause_events[session_id].set()  # Resume the simulation
            simulations[session_id]["status"] = "running"
            simulations[session_id]["paused"] = False
        else:
            return jsonify({"error": "Invalid action"}), 400

        return jsonify({"message": f"Simulation {action}d"})


@app.route("/api/simulations/<session_id>", methods=["DELETE"])
def api_delete_simulation(session_id):
    """Delete a simulation."""
    with simulations_lock:
        if session_id in simulations:
            del simulations[session_id]
            if session_id in pause_events:
                del pause_events[session_id]
            return jsonify({"message": "Simulation deleted"})
        return jsonify({"error": "Simulation not found"}), 404


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Create templates directory if it doesn't exist
    Path("templates").mkdir(exist_ok=True)

    print("\n" + "="*60)
    print("Emotiv Learn Training Simulation - Web Interface")
    print("="*60)
    print("\nStarting server at http://localhost:3000")
    print("\nFeatures:")
    print("  • Modern light mode UI with Inter font")
    print("  • Run multiple simulations in parallel")
    print("  • Pause/Resume simulations anytime")
    print("  • Real-time EEG and behavioral signals")
    print("\nPress Ctrl+C to stop\n")

    app.run(debug=True, threaded=True, host="0.0.0.0", port=3000)

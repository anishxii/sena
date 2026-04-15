"""
Training Simulation Loop
========================
Orchestrates the full Emotiv Learn pipeline for testing/training the RL model.

Flow:
1. User provides initial learning topic prompt
2. LLM generates full lesson plan (split into content steps)
3. For each content step:
   - Present content to simulated user
   - Collect EEG features + behavioral cues + user action
   - RL model chooses content optimization action (currently random)
   - Apply action to modify next content step
4. Log all data for RL training

Usage:
    python train_simulation.py --topic "derivatives and how to calculate them" \
                                --stew-dir ../stew_dataset \
                                --num-steps 10
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from dotenv import load_dotenv

from simulated_user import SimulatedHumanAgent

load_dotenv()


# ── Constants ──────────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
GPT_MODEL = "gpt-4"

# Content optimization actions (RL action space)
CONTENT_ACTIONS = [
    "simplify",         # Reduce complexity, use simpler language
    "deepen",          # Add more depth, technical details
    "add_examples",    # Include concrete examples or case studies
    "use_analogy",     # Explain using analogies or metaphors
    "change_format",   # Switch presentation style (narrative, bullet points, Q&A)
    "maintain",        # Keep current style (neutral action)
]


# ── LLM API Helpers ────────────────────────────────────────────────────────────

def call_openai(system_prompt: str, user_prompt: str, max_tokens: int = 4000) -> str:
    """Call OpenAI API and return the text response."""
    import urllib.request

    payload = json.dumps({
        "model": GPT_MODEL,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
    }).encode()

    req = urllib.request.Request(
        OPENAI_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())

        message_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return message_content.strip() if message_content else ""
    except Exception as e:
        print(f"[ERROR] OpenAI API call failed: {e}")
        raise


def generate_lesson_plan(topic: str, num_steps: int) -> List[Dict[str, str]]:
    """
    Generate a structured lesson plan for the given topic.

    Returns:
        List of dicts, each with {"title": str, "outline": str}
    """
    system_prompt = (
        "You are an expert educational content designer. Create structured lesson plans "
        "that break complex topics into progressive learning steps."
    )

    user_prompt = f"""
Create a {num_steps}-step lesson plan for teaching: "{topic}"

Requirements:
- Each step should build on previous steps
- Progress from fundamentals to more complex concepts
- Each step should be completable in 3-5 minutes of reading
- Return ONLY a JSON array, no other text

Format:
[
  {{"title": "Step 1 title", "outline": "Brief description of what this step covers"}},
  {{"title": "Step 2 title", "outline": "Brief description"}},
  ...
]
"""

    print(f"[INFO] Generating {num_steps}-step lesson plan for: {topic}")
    response = call_openai(system_prompt, user_prompt, max_tokens=2000)

    # Parse JSON response
    try:
        # Extract JSON array from response (handle markdown code blocks)
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()

        plan = json.loads(response)
        print(f"[INFO] Generated lesson plan with {len(plan)} steps")
        return plan
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse lesson plan JSON: {e}")
        print(f"Response was: {response[:500]}")
        raise


def generate_content_step(
    topic: str,
    step_info: Dict[str, str],
    step_number: int,
    total_steps: int,
    previous_content: str = "",
    optimization_action: str = "maintain",
) -> str:
    """
    Generate content for a single lesson step, optionally applying a content optimization action.

    Args:
        topic: Overall learning topic
        step_info: Dict with "title" and "outline" for this step
        step_number: Current step index (1-indexed)
        total_steps: Total number of steps in lesson
        previous_content: Content from previous step (for continuity)
        optimization_action: One of CONTENT_ACTIONS to apply

    Returns:
        Generated content text
    """
    system_prompt = (
        "You are an adaptive educational content generator. Create clear, engaging "
        "educational content tailored to the learner's needs."
    )

    action_instructions = {
        "simplify": "Use simpler language, shorter sentences, and more accessible explanations. Avoid jargon.",
        "deepen": "Include more technical depth, mathematical details, and advanced concepts. Assume higher prior knowledge.",
        "add_examples": "Include 2-3 concrete, worked examples or real-world case studies to illustrate concepts.",
        "use_analogy": "Explain core concepts using clear analogies and metaphors from everyday experience.",
        "change_format": "Switch the presentation format (e.g., if previous was narrative, use bullet points or Q&A style).",
        "maintain": "Continue with the current style and complexity level.",
    }

    action_instruction = action_instructions.get(optimization_action, action_instructions["maintain"])

    context = f"This is step {step_number} of {total_steps} in a lesson on: {topic}\n\n"
    if previous_content:
        context += f"Previous step covered:\n{previous_content[:300]}...\n\n"

    user_prompt = f"""
{context}
Current Step: {step_info['title']}
Outline: {step_info['outline']}

Content Optimization: {action_instruction}

Generate the educational content for this step. Target length: 200-400 words.
Return ONLY the content text, no preamble or metadata.
"""

    print(f"[INFO] Generating content for Step {step_number}/{total_steps} (action: {optimization_action})")
    content = call_openai(system_prompt, user_prompt, max_tokens=2000)
    return content.strip()


# ── Training Loop ──────────────────────────────────────────────────────────────

def run_training_simulation(
    topic: str,
    stew_dir: str,
    num_steps: int = 10,
    subject_id: str = None,
    seed: int = 42,
    output_dir: str = "./training_logs",
):
    """
    Run the full training simulation loop.

    Args:
        topic: Learning topic to generate lesson for
        stew_dir: Path to STEW dataset directory
        num_steps: Number of lesson steps to generate
        subject_id: STEW subject ID (e.g., "sub01"), or None for random
        seed: Random seed for reproducibility
        output_dir: Directory to save training logs
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"session_{timestamp}"

    print("\n" + "="*80)
    print("EMOTIV LEARN — TRAINING SIMULATION")
    print("="*80)
    print(f"Topic: {topic}")
    print(f"Steps: {num_steps}")
    print(f"Subject: {subject_id or 'random'}")
    print(f"Session ID: {session_id}")
    print("="*80 + "\n")

    # Initialize simulated user
    print("[1/4] Initializing simulated user agent...")
    agent = SimulatedHumanAgent(
        stew_dir=stew_dir,
        subject_id=subject_id,
        seed=seed,
        loop_epochs=True,
    )
    print(f"      Loaded: {agent}\n")

    # Generate lesson plan
    print("[2/4] Generating lesson plan...")
    lesson_plan = generate_lesson_plan(topic, num_steps)
    print()

    # Training data collection
    training_data = {
        "session_id": session_id,
        "topic": topic,
        "subject_id": agent.subject_id,
        "subject_rating": float(agent.rating),
        "lesson_plan": lesson_plan,
        "steps": [],
    }

    print("[3/4] Running simulation loop...\n")

    previous_content = ""
    current_action = "maintain"  # Start with neutral action

    for i, step_info in enumerate(lesson_plan):
        step_num = i + 1
        print(f"\n{'─'*80}")
        print(f"STEP {step_num}/{num_steps}: {step_info['title']}")
        print(f"{'─'*80}")

        # Generate content with current optimization action
        content = generate_content_step(
            topic=topic,
            step_info=step_info,
            step_number=step_num,
            total_steps=num_steps,
            previous_content=previous_content,
            optimization_action=current_action,
        )

        print(f"\nContent preview ({len(content)} chars):")
        print(f"  {content[:150]}...")

        # Simulated user processes content
        print(f"\nSimulated user reading...")
        start_time = time.time()
        user_response = agent.step(content)
        elapsed = time.time() - start_time

        print(f"\nUser response (processed in {elapsed:.2f}s):")
        print(f"  Cognitive load:  {user_response['cognitive_load']:.3f}")
        print(f"  Time on chunk:   {user_response['behavioral_cues']['time_on_chunk']}s")
        print(f"  Scroll rate:     {user_response['behavioral_cues']['scroll_rate']:.3f}")
        print(f"  Reread count:    {user_response['behavioral_cues']['reread_count']}")
        print(f"  Epochs consumed: {user_response['epochs_consumed']}")
        print(f"  Next action:     {user_response['next_action']}")
        if user_response['next_prompt']:
            print(f"  User message:    \"{user_response['next_prompt']}\"")

        # RL model would go here — for now, sample random action
        next_action = rng.choice(CONTENT_ACTIONS)
        print(f"\n  RL action:       {next_action} (random)")

        # Log this step
        step_data = {
            "step_number": step_num,
            "step_title": step_info['title'],
            "content": content,
            "content_length": len(content),
            "optimization_action_applied": current_action,
            "eeg_features": user_response['eeg_features'].tolist(),
            "behavioral_cues": user_response['behavioral_cues'],
            "cognitive_load": user_response['cognitive_load'],
            "epochs_consumed": user_response['epochs_consumed'],
            "user_action": user_response['next_action'],
            "user_prompt": user_response['next_prompt'],
            "rl_action": next_action,
        }
        training_data['steps'].append(step_data)

        # Update for next iteration
        previous_content = content
        current_action = next_action

    # Save training data
    print(f"\n[4/4] Saving training data...")
    output_file = output_path / f"{session_id}.json"
    with open(output_file, 'w') as f:
        json.dump(training_data, f, indent=2)

    print(f"      Saved to: {output_file}")

    # Print summary statistics
    print(f"\n{'='*80}")
    print("SIMULATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total steps:           {len(training_data['steps'])}")

    avg_load = np.mean([s['cognitive_load'] for s in training_data['steps']])
    avg_time = np.mean([s['behavioral_cues']['time_on_chunk'] for s in training_data['steps']])

    print(f"Avg cognitive load:    {avg_load:.3f}")
    print(f"Avg time per step:     {avg_time:.1f}s")

    action_counts = {}
    for s in training_data['steps']:
        action = s['user_action']
        action_counts[action] = action_counts.get(action, 0) + 1

    print(f"\nUser action distribution:")
    for action, count in sorted(action_counts.items()):
        print(f"  {action:10s}: {count:2d} ({100*count/num_steps:.1f}%)")

    print(f"\nTraining log: {output_file}")
    print(f"{'='*80}\n")

    return training_data


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run Emotiv Learn training simulation with simulated user agent"
    )
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="Learning topic to generate lesson for (e.g., 'derivatives and how to calculate them')",
    )
    parser.add_argument(
        "--stew-dir",
        type=str,
        default="../stew_dataset",
        help="Path to STEW dataset directory (default: ../stew_dataset)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Number of lesson steps to generate (default: 10)",
    )
    parser.add_argument(
        "--subject-id",
        type=str,
        default=None,
        help="STEW subject ID (e.g., 'sub01'), or None for random (default: None)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./training_logs",
        help="Directory to save training logs (default: ./training_logs)",
    )

    args = parser.parse_args()

    # Validate API key
    if not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY not found in environment")
        print("        Set it in .env file or export OPENAI_API_KEY=<your-key>")
        return 1

    # Run simulation
    try:
        run_training_simulation(
            topic=args.topic,
            stew_dir=args.stew_dir,
            num_steps=args.num_steps,
            subject_id=args.subject_id,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        return 0
    except Exception as e:
        print(f"\n[ERROR] Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

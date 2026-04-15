# Web Interface Guide

## Overview

The web interface provides a ChatGPT-like experience for running and monitoring multiple training simulations simultaneously.

## Starting the Interface

```bash
cd train
./start_web.sh
```

Then open **http://localhost:3000** in your browser.

## New Features ✨

- **⏸️ Pause/Resume** - Click the pause button in the header to pause a running simulation. Click resume to continue from where you left off.
- **Modern Light Mode** - Clean, modern design with light backgrounds and the beautiful Inter font
- **Gradient Accents** - Purple gradient buttons and progress bars
- **Smooth Animations** - Polished hover effects and transitions

## Interface Layout

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌──────────────┐  ┌───────────────────────────────────┐  │
│  │   SIDEBAR    │  │        MAIN CONTENT AREA         │  │
│  │              │  │                                   │  │
│  │ + New Sim    │  │  Topic: Derivatives               │  │
│  │              │  │  Subject: sub01 | Rating: 6.0/9   │  │
│  │ ─────────    │  │  Steps: 3/8 | Status: running    │  │
│  │              │  │                                   │  │
│  │ ○ Deriv...   │  │  ┌─────────────────────────────┐ │  │
│  │   running    │  │  │ AI | Step 1: Introduction   │ │  │
│  │   sub01      │  │  │ ───────────────────────────│ │  │
│  │   [▓▓▓░░]    │  │  │ Content text here...        │ │  │
│  │              │  │  │                             │ │  │
│  │ ○ Neural...  │  │  │ ┌──────┬──────┬──────┐    │ │  │
│  │   completed  │  │  │ │ Load │ Time │Scroll│    │ │  │
│  │   sub05      │  │  │ │ 45%  │ 52s  │0.823 │    │ │  │
│  │              │  │  │ └──────┴──────┴──────┘    │ │  │
│  │              │  │  │ EEG: [0.12, 0.34, ...]    │ │  │
│  │              │  │  └─────────────────────────────┘ │  │
│  │              │  │                                   │  │
│  │              │  │  ┌─────────────────────────────┐ │  │
│  │              │  │  │ U | Student Response       │ │  │
│  │              │  │  │ ───────────────────────────│ │  │
│  │              │  │  │ "Can you explain that      │ │  │
│  │              │  │  │  differently?"              │ │  │
│  │              │  │  └─────────────────────────────┘ │  │
│  │              │  │                                   │  │
│  └──────────────┘  └───────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Features

### 1. **Sidebar - Simulation Management**

- **+ New Simulation** button - Start a new simulation
- **Simulation List** - Shows all active/completed simulations with:
  - Topic name
  - Status badge (running, completed, error)
  - Subject ID
  - Progress (current step / total steps)
  - Progress bar for running simulations
  - Click any simulation to view details

### 2. **Main Content Area - Live Simulation View**

#### Header
Shows current simulation metadata:
- Topic name
- Subject ID and workload rating
- Progress (steps completed)
- Current status
- **⏸️ Pause button** (appears when simulation is running)
- **▶️ Resume button** (appears when simulation is paused)

#### Content Steps (Chat-like Display)

Each step shows:

**AI Message:**
- Step number and title
- Full content text
- **Signal Dashboard** with:
  - 🔴 **Cognitive Load** - Color-coded (red=high, yellow=medium, green=low)
  - ⏱️ **Time on Chunk** - How long the user spent reading (seconds)
  - 📊 **Scroll Rate** - Engagement indicator (0-1)
  - 🔄 **Reread Count** - How many times content was re-read
  - 🧠 **Epochs Consumed** - Number of EEG epochs used
- **EEG Features Preview** - First 6 values of 62-dim vector

**Student Response (if any):**
- User action (continue/clarify/branch)
- LLM-generated student prompt/question

**RL Action:**
- Shows which content optimization action was chosen for the next step

### 3. **New Simulation Modal**

Click "+ New Simulation" to open a form with:

- **Learning Topic** (required) - e.g., "derivatives and how to calculate them"
- **Number of Steps** (1-20) - Default: 8
- **Subject ID** - Dropdown with all 48 available subjects, or "Random"
- **Start Simulation** / **Cancel** buttons

### 4. **Real-time Updates**

The interface auto-refreshes every 2 seconds to show:
- New content steps as they're generated
- Updated progress bars
- Status changes
- Multiple simulations running in parallel

## Color Coding

### Cognitive Load Indicators
- 🟢 **Green (Low)** - Load < 40% - User is comfortable
- 🟡 **Yellow (Medium)** - Load 40-70% - Moderate engagement
- 🔴 **Red (High)** - Load > 70% - User is struggling

### Status Badges
- 🟢 **Running** - Simulation in progress
- 🟠 **Paused** - Simulation paused (can be resumed)
- ⚪ **Completed** - Simulation finished
- 🟣 **Starting** - Initializing
- 🔴 **Error** - Something went wrong

## Usage Examples

### Example 1: Single Simulation

1. Click "+ New Simulation"
2. Enter topic: "how neural networks learn"
3. Steps: 10
4. Subject: Random
5. Click "Start Simulation"
6. Watch as each step appears with real-time EEG signals

### Example 2: Multiple Simulations

1. Start simulation #1 with subject sub01 on topic "calculus"
2. While #1 is running, start simulation #2 with subject sub05 on topic "physics"
3. Click between them in the sidebar to monitor each
4. Both run independently in parallel

### Example 3: Pause and Resume

Pause a simulation to free up API quota:

1. Start simulation with 10 steps
2. After step 3, click "⏸ Pause" button
3. Simulation freezes (status changes to "paused")
4. Work on something else or start other simulations
5. Come back and click "▶ Resume"
6. Simulation continues from step 4

### Example 4: Subject Comparison

Compare how different subjects respond to the same content:

1. Start simulation with sub01, topic "derivatives", 8 steps
2. Start simulation with sub15, topic "derivatives", 8 steps
3. Start simulation with sub30, topic "derivatives", 8 steps
4. Switch between them to compare cognitive load patterns

## API Endpoints

The web interface uses these REST endpoints:

- `GET /` - Main page
- `GET /api/subjects` - List available STEW subjects
- `GET /api/simulations` - List all simulations
- `GET /api/simulations/<id>` - Get simulation details
- `GET /api/simulations/<id>/steps` - Get all steps for a simulation
- `POST /api/simulations` - Start new simulation
- `POST /api/simulations/<id>/pause` - Pause or resume a simulation (body: `{"action": "pause"}` or `{"action": "resume"}`)
- `DELETE /api/simulations/<id>` - Delete a simulation

## Keyboard Shortcuts

- **ESC** - Close modal

## Browser Compatibility

Tested on:
- Chrome/Edge (recommended)
- Firefox
- Safari

## Performance Notes

- Each simulation runs in a separate thread
- Up to ~10 simultaneous simulations recommended
- Page auto-refreshes every 2 seconds
- Large simulations (15+ steps) may take 5-10 minutes

## Troubleshooting

### Port Already in Use
```bash
# Kill existing Flask process
pkill -f web_interface.py

# Or use a different port
python web_interface.py
# Then edit app.run(port=5001)
```

### Simulations Not Updating
- Check browser console for errors
- Refresh the page
- Verify OPENAI_API_KEY is set in `.env`

### Can't Start New Simulation
- Check that STEW dataset is in `../stew_dataset/`
- Verify API key is valid
- Check server logs for errors

## Data Storage

- Simulations are stored **in memory only** during server runtime
- When server stops, all simulation data is lost
- To persist data, use the CLI mode which saves JSON logs

## Next Steps

After running simulations in the web interface:

1. **Save results** - Copy interesting simulation IDs from the interface
2. **Run CLI mode** to generate permanent JSON logs
3. **Analyze logs** with `analyze_training_logs.py`
4. **Train RL model** using the collected data

#!/bin/bash
# Start the web interface for training simulations

echo "================================"
echo "Emotiv Learn - Web Interface"
echo "================================"
echo ""

# Check if .env exists
if [ ! -f ../.env ]; then
    echo "ERROR: ../.env file not found!"
    echo "Create .env in parent directory with:"
    echo "  OPENAI_API_KEY=your_key_here"
    exit 1
fi

# Check if STEW dataset exists
if [ ! -d ../stew_dataset ]; then
    echo "ERROR: ../stew_dataset directory not found!"
    echo "Place STEW dataset files in ../stew_dataset/"
    exit 1
fi

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "Installing Flask..."
    pip install flask
fi

echo "Starting web server..."
echo ""
echo "Open http://localhost:3000 in your browser"
echo ""
echo "Features:"
echo "  • Modern light mode UI with Inter font"
echo "  • Run multiple simulations in parallel"
echo "  • Pause/Resume simulations anytime"
echo "  • Real-time EEG and behavioral signals"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python web_interface.py

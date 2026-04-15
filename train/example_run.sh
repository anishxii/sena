#!/bin/bash
# Quick example to test the training simulation setup

echo "================================"
echo "Emotiv Learn - Training Simulation Test"
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

echo "Running training simulation..."
echo "Topic: Introduction to derivatives"
echo "Steps: 5 (short test)"
echo ""

python train_simulation.py \
    --topic "what are derivatives and how to calculate them" \
    --stew-dir ../stew_dataset \
    --num-steps 5 \
    --subject-id sub01 \
    --seed 42 \
    --output-dir ./training_logs

if [ $? -eq 0 ]; then
    echo ""
    echo "================================"
    echo "SUCCESS! Training simulation completed"
    echo "================================"
    echo ""
    echo "Next steps:"
    echo "1. Check the log file in ./training_logs/"
    echo "2. Run analysis:"
    echo "   python analyze_training_logs.py --log-dir ./training_logs"
    echo "3. Try different topics and longer sessions"
else
    echo ""
    echo "ERROR: Simulation failed. Check error messages above."
    exit 1
fi

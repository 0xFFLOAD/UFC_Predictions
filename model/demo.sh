#!/bin/bash
# Quick demo of UFC prediction model

set -e

echo "╔════════════════════════════════════════════════════╗"
echo "║  UFC Winner Prediction Neural Network Demo        ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

# Check if data exists
if [ ! -f "../data/ufc_fights_full_with_odds.csv" ]; then
    echo "❌ Error: UFC dataset not found!"
    echo "   Expected: ../data/ufc_fights_full_with_odds.csv"
    exit 1
fi

echo "✅ UFC dataset found"

# Build if needed
if [ ! -f "ufc_nn" ]; then
    echo "🔨 Building model..."
    make > /dev/null 2>&1
    echo "✅ Model built successfully"
else
    echo "✅ Model executable found"
fi

# Check if trained model exists
if [ -f "ufc_model.bin" ]; then
    echo "✅ Trained model found (ufc_model.bin)"
    echo ""
    echo "Run the following commands:"
    echo ""
    echo "  make predict    # Interactive prediction mode"
    echo "  make info       # Show model information"
    echo "  make train      # Retrain from scratch"
    echo "  python3 evaluate.py  # Evaluate model performance"
    echo ""
else
    echo "⚠️  No trained model found"
    echo ""
    echo "To train the model, run:"
    echo "  make train"
    echo ""
    echo "This will:"
    echo "  • Load 7,340+ UFC fights from historical data"
    echo "  • Train a 14→64→32→1 neural network"
    echo "  • Save the trained model to ufc_model.bin"
    echo "  • Take ~1-5 minutes depending on your CPU"
    echo ""
    read -p "Train now? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        make train
    else
        echo "Skipping training. Run 'make train' when ready."
    fi
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo "Model Architecture:"
echo "  Input:  14 features (fighter stat deltas)"
echo "  Hidden: 64 neurons (tanh) → 32 neurons (tanh)"
echo "  Output: P(Fighter 1 wins)"
echo ""
echo "Features include:"
echo "  • Physical: height, reach, age, weight deltas"
echo "  • Striking: accuracy, output, absorbed, defense"
echo "  • Grappling: takedowns, submissions"
echo "  • Derived: striking advantage, grappling score"
echo "═══════════════════════════════════════════════════"

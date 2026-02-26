#!/bin/bash
# Quick demo of Tesla forecast model

set -e

echo "╔════════════════════════════════════════════════════╗"
echo "║  Tesla Stock Forecast Model Demo                  ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

DATASET_PRIMARY="../data/TSLA.csv"
DATASET_FALLBACK="data/TSLA.csv"
DATASET=""

if [ -f "$DATASET_PRIMARY" ]; then
    DATASET="$DATASET_PRIMARY"
elif [ -f "$DATASET_FALLBACK" ]; then
    DATASET="$DATASET_FALLBACK"
fi

# Check if data exists
if [ -z "$DATASET" ]; then
    echo "❌ Error: Tesla dataset not found!"
    echo "   Checked: $DATASET_PRIMARY and $DATASET_FALLBACK"
    exit 1
fi

echo "✅ Tesla dataset found: $DATASET"

# Build if needed
if [ ! -f "tsla_nn" ]; then
    echo "🔨 Building model..."
    make > /dev/null 2>&1
    echo "✅ Model built successfully"
else
    echo "✅ Model executable found"
fi

# Check if trained model exists
if [ -f "tsla_model.bin" ]; then
    echo "✅ Trained model found (tsla_model.bin)"
    echo ""
    echo "Run the following commands:"
    echo ""
    echo "  make predict    # Next-period forecast"
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
    echo "  • Load TSLA monthly OHLCV history"
    echo "  • Estimate baseline return + volatility"
    echo "  • Save model to tsla_model.bin"
    echo "  • Finish in seconds"
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
echo "Model Output:"
echo "  • Forecast date"
echo "  • Predicted close"
echo "  • Expected return (%)"
echo "  • Bullish probability (%)"
echo "═══════════════════════════════════════════════════"

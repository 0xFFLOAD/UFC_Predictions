#!/bin/bash
# Quick start guide for Tesla forecast model

echo "╔══════════════════════════════════════════════════════════╗"
echo "║         TESLA FORECAST QUICK START                      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "This guide walks you through training and using the"
echo "Tesla stock forecast model."
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

echo "📍 Current directory: $(pwd)"
echo ""

# Step 1: Verify data
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Verify Tesla Dataset"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -n "$DATASET" ]; then
    lines=$(wc -l < "$DATASET")
    size=$(ls -lh "$DATASET" | awk '{print $5}')
    echo "✅ Dataset found: $DATASET"
    echo "   Lines: $lines"
    echo "   Size: $size"
else
    echo "❌ Dataset not found!"
    echo "   Checked: $DATASET_PRIMARY and $DATASET_FALLBACK"
    exit 1
fi

echo ""

# Step 2: Build
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Build Neural Network"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "tsla_nn" ]; then
    echo "✅ Executable already built: tsla_nn"
else
    echo "🔨 Building..."
    if make > /dev/null 2>&1; then
        echo "✅ Build successful!"
    else
        echo "❌ Build failed. Check for compilation errors."
        exit 1
    fi
fi

echo ""

# Step 3: Train
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Train Model"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "tsla_model.bin" ]; then
    size=$(ls -lh "tsla_model.bin" | awk '{print $5}')
    echo "✅ Trained model exists: tsla_model.bin ($size)"
    echo ""
    read -p "Retrain from scratch? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing model."
    else
        echo ""
        echo "Training baseline on historical TSLA closes..."
        echo ""
        make train
    fi
else
    echo "No trained model found. Training now..."
    echo "This takes a few seconds..."
    echo ""
    read -p "Continue? (Y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        echo "Training cancelled. Run 'make train' when ready."
        exit 0
    fi
    echo ""
    make train
fi

echo ""

# Step 4: Usage
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Usage examples:"
echo ""
echo "  1. Forecast next period:"
echo "     make predict"
echo ""
echo "  2. Show model info:"
echo "     make info"
echo ""
echo "  3. Evaluate model:"
echo "     python3 evaluate.py"
echo ""
echo "  4. Retrain:"
echo "     make train"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Example prediction output:"
echo ""
echo "  \$ make predict"
echo "  >>> Predicted close: ..."
echo "  >>> Expected return: ...%"
echo "  >>> Bullish probability: ...%"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

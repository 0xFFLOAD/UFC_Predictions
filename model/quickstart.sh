#!/bin/bash
# Quick start guide for UFC prediction model

echo "╔══════════════════════════════════════════════════════════╗"
echo "║         UFC NEURAL NETWORK QUICK START                  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "This guide walks you through training and using the UFC"
echo "winner prediction neural network."
echo ""

cd "$(dirname "$0")"

echo "📍 Current directory: $(pwd)"
echo ""

# Step 1: Verify data
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Verify UFC Dataset"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "../data/ufc_fights_full_with_odds.csv" ]; then
    lines=$(wc -l < "../data/ufc_fights_full_with_odds.csv")
    size=$(ls -lh "../data/ufc_fights_full_with_odds.csv" | awk '{print $5}')
    echo "✅ Dataset found: ../data/ufc_fights_full_with_odds.csv"
    echo "   Lines: $lines"
    echo "   Size: $size"
else
    echo "❌ Dataset not found!"
    echo "   Please ensure ../data/ufc_fights_full_with_odds.csv exists"
    exit 1
fi

echo ""

# Step 2: Build
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Build Neural Network"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "ufc_nn" ]; then
    echo "✅ Executable already built: ufc_nn"
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

if [ -f "ufc_model.bin" ]; then
    size=$(ls -lh "ufc_model.bin" | awk '{print $5}')
    echo "✅ Trained model exists: ufc_model.bin ($size)"
    echo ""
    read -p "Retrain from scratch? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing model."
    else
        echo ""
        echo "Training on 7,000+ UFC fights..."
        echo "This may take 1-5 minutes..."
        echo ""
        make train
    fi
else
    echo "No trained model found. Training now..."
    echo "This will take 1-5 minutes..."
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
echo "  1. Interactive prediction:"
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
echo "Example prediction session:"
echo ""
echo "  \$ make predict"
echo "  >>> Fighter 1 stats: 180 185 28 4.5 2.1"
echo "  >>> Fighter 2 stats: 175 178 32 3.8 1.5"
echo "  >>> Output: Fighter 1 win probability: 67.32%"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

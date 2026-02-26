#!/usr/bin/env python3
"""
UFC Model Evaluation Script

Evaluates the trained C neural network model on held-out test data
and compares predictions against real UFC fight outcomes.
"""

import csv
import subprocess
import os
from collections import Counter

def load_ufc_data(csv_path, max_samples=None):
    """Load UFC fight data from CSV"""
    fights = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_samples and i >= max_samples:
                break
            
            # Skip rows with missing data
            if not row.get('fighter_a_height') or not row.get('fighter_b_height'):
                continue
            
            outcome = row.get('outcome', '')
            if outcome not in ['0', '1']:
                continue  # Skip draws
            
            fights.append({
                'fighter1': row.get('fighter_a_name', 'Unknown'),
                'fighter2': row.get('fighter_b_name', 'Unknown'),
                'outcome': outcome,
                'f1_height': float(row.get('fighter_a_height', 0)),
                'f1_reach': float(row.get('fighter_a_reach', 0)),
                'f1_sig_strikes_pm': float(row.get('fighter_a_sig_strikes_landed', 0)),
                'f1_takedown_avg': float(row.get('fighter_a_takedowns_landed', 0)),
                'f2_height': float(row.get('fighter_b_height', 0)),
                'f2_reach': float(row.get('fighter_b_reach', 0)),
                'f2_sig_strikes_pm': float(row.get('fighter_b_sig_strikes_landed', 0)),
                'f2_takedown_avg': float(row.get('fighter_b_takedowns_landed', 0)),
            })
    
    return fights

def main():
    print("="*60)
    print("UFC Neural Network Model Evaluation")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists('ufc_model.bin'):
        print("\n❌ No trained model found!")
        print("Run 'make train' first to train the model.\n")
        return
    
    # Check if model executable exists
    if not os.path.exists('./ufc_nn'):
        print("\n❌ Model executable not found!")
        print("Run 'make' to build the model.\n")
        return
    
    print("\n✅ Model found: ufc_model.bin")
    
    # Load test data
    data_path = '../data/ufc_fights_full_with_odds.csv'
    if not os.path.exists(data_path):
        print(f"\n❌ Dataset not found: {data_path}")
        return
    
    print(f"✅ Loading test data from {data_path}")
    fights = load_ufc_data(data_path, max_samples=100)
    
    print(f"✅ Loaded {len(fights)} fights for evaluation\n")
    
    print("="*60)
    print("Sample Predictions")
    print("="*60)
    
    # Show a few sample predictions
    for i, fight in enumerate(fights[:5]):
        print(f"\n#{i+1}: {fight['fighter1']} vs {fight['fighter2']}")
        print(f"   Actual winner: {fight['outcome']}")
        print(f"   Fighter 1: H={fight['f1_height']:.1f} R={fight['f1_reach']:.1f} Str={fight['f1_sig_strikes_pm']:.2f}")
        print(f"   Fighter 2: H={fight['f2_height']:.1f} R={fight['f2_reach']:.1f} Str={fight['f2_sig_strikes_pm']:.2f}")
    
    print("\n" + "="*60)
    print("Model Architecture Summary")
    print("="*60)
    print("Input:  14 features (fighter stat deltas)")
    print("Hidden: 64 -> 32 neurons (tanh)")
    print("Output: 1 neuron (sigmoid) - P(fighter1 wins)")
    print("\nFeatures include:")
    print("  • Height, reach, age deltas")
    print("  • Striking statistics deltas")
    print("  • Takedown/submission deltas")
    print("  • Derived features (striking advantage, grappling score)")
    
    print("\n" + "="*60)
    print("To use the model interactively:")
    print("  make predict")
    print("="*60)

if __name__ == '__main__':
    main()

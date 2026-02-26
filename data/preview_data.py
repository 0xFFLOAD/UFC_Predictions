#!/usr/bin/env python3
"""
UFC Data Preview Script
Analyze and summarize the downloaded UFC datasets
"""

import csv
import os
from collections import Counter
from datetime import datetime

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def preview_csv(filename, max_rows=5):
    """Preview first few rows and basic stats of a CSV file"""
    print(f"\n{'='*80}")
    print(f"Dataset: {filename}")
    print(f"{'='*80}")
    
    # Construct full path relative to script location
    filepath = os.path.join(SCRIPT_DIR, filename)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            if not rows:
                print("⚠️  Empty dataset")
                return
            
            total_rows = len(rows)
            columns = list(rows[0].keys())
            
            print(f"\n📊 Summary:")
            print(f"   • Total rows: {total_rows:,}")
            print(f"   • Total columns: {len(columns)}")
            
            # Preview columns
            print(f"\n📋 Columns ({len(columns)}):")
            for i, col in enumerate(columns[:20], 1):
                print(f"   {i:2d}. {col}")
            if len(columns) > 20:
                print(f"   ... and {len(columns) - 20} more columns")
            
            # Preview data
            print(f"\n🔍 First {min(max_rows, total_rows)} rows:")
            for i, row in enumerate(rows[:max_rows], 1):
                print(f"\n   Row {i}:")
                for key, value in list(row.items())[:10]:
                    display_val = value[:60] if len(value) > 60 else value
                    print(f"      {key}: {display_val}")
                if len(row) > 10:
                    print(f"      ... and {len(row) - 10} more fields")
            
            return rows, columns
            
    except Exception as e:
        print(f"❌ Error reading {filename}: {e}")
        return None, None


def analyze_complete_dataset(rows):
    """Analyze the comprehensive UFC dataset"""
    if not rows:
        return
    
    print(f"\n{'='*80}")
    print("Deep Analysis: ufc_complete_dataset.csv")
    print(f"{'='*80}")
    
    # Date range
    dates = [row['event_date'] for row in rows if row.get('event_date')]
    if dates:
        print(f"\n📅 Date Range: {min(dates)} to {max(dates)}")
    
    # Weight classes
    weight_classes = Counter(row['weight_class'] for row in rows if row.get('weight_class'))
    print(f"\n⚖️  Weight Classes (top 10):")
    for wc, count in weight_classes.most_common(10):
        print(f"   • {wc}: {count:,} fights")
    
    # Outcomes
    outcomes = Counter(row['outcome'] for row in rows if row.get('outcome'))
    print(f"\n🎯 Outcomes:")
    for outcome, count in outcomes.most_common(5):
        print(f"   • {outcome}: {count:,}")
    
    # Methods
    methods = Counter(row['method'] for row in rows if row.get('method'))
    print(f"\n🥊 Victory Methods (top 10):")
    for method, count in methods.most_common(10):
        print(f"   • {method}: {count:,}")
    
    # Betting data availability
    with_odds = sum(1 for row in rows if row.get('favourite_odds') and row['favourite_odds'].strip())
    print(f"\n💰 Betting Data:")
    print(f"   • Fights with odds: {with_odds:,} ({with_odds/len(rows)*100:.1f}%)")
    print(f"   • Fights without odds: {len(rows)-with_odds:,}")
    
    # Unique fighters
    fighters = set()
    for row in rows:
        if row.get('fighter1'):
            fighters.add(row['fighter1'])
        if row.get('fighter2'):
            fighters.add(row['fighter2'])
    print(f"\n👤 Unique Fighters: {len(fighters):,}")


def main():
    print("\n" + "="*80)
    print("🥊 UFC DATA PREVIEW (Class-Filtered)")
    print("="*80)
    
    datasets = [
        'ufc_complete_dataset.csv'
    ]
    
    all_data = {}
    
    for dataset in datasets:
        rows, cols = preview_csv(dataset, max_rows=2)
        if rows:
            all_data[dataset] = (rows, cols)
    
    # Deep analysis for complete dataset
    if 'ufc_complete_dataset.csv' in all_data:
        analyze_complete_dataset(all_data['ufc_complete_dataset.csv'][0])
    
    print(f"\n{'='*80}")
    print("✅ Data preview complete!")
    print("="*80)
    print("\n💡 Next steps:")
    print("   • Load data with pandas: df = pd.read_csv('data/ufc_complete_dataset.csv')")
    print("   • Check data/README.md for detailed field descriptions")
    print("   • Explore fighter statistics, betting trends, and match outcomes")
    print()


if __name__ == "__main__":
    main()

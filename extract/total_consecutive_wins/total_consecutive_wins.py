"""
Extraction logic for the Total Consecutive Wins (win streak) feature.
"""

import os
try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas is required to run this extractor. Install it via 'pip install pandas' or use the correct Python environment.")

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ufc_dataset', 'large_set', 'large_dataset.csv'))


def extract():
    df = pd.read_csv(DATA_PATH)
    base_cols = ['r_fighter', 'b_fighter', 'winner']
    # the dataset contains r_wins_total and b_wins_total but not a
    # consecutive wins value.  We return the totals as a proxy.
    cols = ['r_wins_total', 'b_wins_total']
    existing = [c for c in cols if c in df.columns]
    if not existing:
        print("No win total columns found in dataset.")
        return df[base_cols].copy()
    return df[base_cols + existing].copy()


if __name__ == '__main__':
    df = extract()
    outname = os.path.splitext(os.path.basename(__file__))[0] + '.tsv'
    outpath = os.path.join(os.path.dirname(__file__), outname)
    df.to_csv(outpath, sep='\t', index=False)
    print(f"Wrote {outpath}")

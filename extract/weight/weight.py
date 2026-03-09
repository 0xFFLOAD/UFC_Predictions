"""
Extraction logic for the Weight feature (absolute weights of fighters).
"""

import os
try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas is required to run this extractor. Install it via 'pip install pandas' or use the correct Python environment.")

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ufc_dataset', 'large_set', 'large_dataset.csv'))


def extract():
    df = pd.read_csv(DATA_PATH)
    # include weight_class so that downstream merges can restrict to
    # fights in the same division.  without it we were joining fighters
    # across classes, which produced absurd weight deltas (>250kg).
    base_cols = ['r_fighter', 'b_fighter', 'winner']
    if 'weight_class' in df.columns:
        base_cols.append('weight_class')
    cols = ['r_weight', 'b_weight', 'weight_diff']
    existing = [c for c in cols if c in df.columns]
    if not existing:
        print("No relevant columns found for Weight in dataset.")
        return df[base_cols].copy()
    return df[base_cols + existing].copy()


if __name__ == '__main__':
    df = extract()
    outname = os.path.splitext(os.path.basename(__file__))[0] + '.tsv'
    outpath = os.path.join(os.path.dirname(__file__), outname)
    df.to_csv(outpath, sep='\t', index=False)
    print(f"Wrote {outpath}")

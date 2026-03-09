"""
Extraction logic for the Age feature (absolute ages of the fighters).
"""

import os
try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas is required to run this extractor. Install it via 'pip install pandas' or use the correct Python environment.")

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ufc_dataset', 'large_set', 'large_dataset.csv'))


def extract():
    df = pd.read_csv(DATA_PATH)
    base_cols = ['r_fighter', 'b_fighter', 'winner', 'weight_class']
    # only include absolute ages here, delta is handled by age_delta
    cols = ['r_age', 'b_age']
    existing = [c for c in cols if c in df.columns]
    if not existing:
        print("No relevant columns found for Age in dataset.")
        return df[base_cols].copy()
    # keep weight_class to allow grouping by class later
    return df[base_cols + existing].copy()


if __name__ == '__main__':
    df = extract()
    # write full combined file
    outname = os.path.splitext(os.path.basename(__file__))[0] + '.tsv'
    outpath = os.path.join(os.path.dirname(__file__), outname)
    df.to_csv(outpath, sep='\t', index=False)
    print(f"Wrote {outpath}")
    # also split by weight class
    if 'weight_class' in df.columns:
        for wc, sub in df.groupby('weight_class'):
            safe = wc.replace(' ', '_').replace('/', '_')
            subpath = os.path.join(os.path.dirname(__file__), f"age_{safe}.tsv")
            sub.to_csv(subpath, sep='\t', index=False)
            print(f"Wrote {subpath} ({len(sub)} rows)")

"""
Extraction logic for the Height Delta feature.
"""

import os
try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas is required to run this extractor. Install it via 'pip install pandas' or use the correct Python environment.")

# compute path relative to workspace root (two levels above current file)
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ufc_dataset', 'large_set', 'large_dataset.csv'))


def extract():
    df = pd.read_csv(DATA_PATH)
    base_cols = ['r_fighter', 'b_fighter', 'winner']
    # 'height_diff' column represents difference in height between fighters (cm).
    cols = ['height_diff']
    existing = [c for c in cols if c in df.columns]
    if not existing:
        print("No relevant columns found for Height Delta in dataset.")
        return df[base_cols].copy()
    return df[base_cols + existing].copy()


if __name__ == '__main__':
    df = extract()
    outname = os.path.splitext(os.path.basename(__file__))[0] + '.tsv'
    outpath = os.path.join(os.path.dirname(__file__), outname)
    df.to_csv(outpath, sep='\t', index=False)
    print(f"Wrote {outpath}")

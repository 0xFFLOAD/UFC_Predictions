"""
Extraction logic for the Previous Wins Against Opponent feature.

If historical matchup data were present we could extract the count
of prior wins versus the same opponent.  The large dataset is
isolated to single events and doesn't include matchup histories,
so this extractor returns empty.
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
    print("No previous-opponent win history available in dataset.")
    return df[base_cols].copy()


if __name__ == '__main__':
    df = extract()
    outname = os.path.splitext(os.path.basename(__file__))[0] + '.tsv'
    outpath = os.path.join(os.path.dirname(__file__), outname)
    df.to_csv(outpath, sep='\t', index=False)
    print(f"Wrote {outpath}")

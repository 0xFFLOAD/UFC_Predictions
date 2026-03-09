"""
Extraction logic for the "Unprepared for Fight" binary feature.

This corresponds to short-notice fights.  The large dataset does
not appear to contain an explicit column for this information, so
the extractor will simply indicate that no data is available.
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
    # placeholder - dataset lacks a short-notice boolean
    print("Feature not present in dataset; returning empty DataFrame.")
    return df[base_cols].copy()


if __name__ == '__main__':
    df = extract()
    outname = os.path.splitext(os.path.basename(__file__))[0] + '.tsv'
    outpath = os.path.join(os.path.dirname(__file__), outname)
    df.to_csv(outpath, sep='\t', index=False)
    print(f"Wrote {outpath}")

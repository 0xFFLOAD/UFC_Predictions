"""
Extraction logic for the Striking Advantage composite feature.

This feature is defined in the README as the net striking
effectiveness (output - absorbed).  The large dataset does not
contain a column with this name, so we compute a proxy using
available difference columns.
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
    # try to compute advantage from significant strikes and absorbed proxies
    if 'sig_str_diff' in df.columns:
        # use sig_str_att_diff as an approximation for absorbed
        absorbed = df['sig_str_att_diff'] if 'sig_str_att_diff' in df.columns else 0
        df['striking_advantage'] = df['sig_str_diff'] - absorbed
        return df[base_cols + ['striking_advantage']].copy()
    else:
        print("Cannot compute striking advantage - required columns missing.")
        return df[base_cols].copy()


if __name__ == '__main__':
    df = extract()
    outname = os.path.splitext(os.path.basename(__file__))[0] + '.tsv'
    outpath = os.path.join(os.path.dirname(__file__), outname)
    df.to_csv(outpath, sep='\t', index=False)
    print(f"Wrote {outpath}")

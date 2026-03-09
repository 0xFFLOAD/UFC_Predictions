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
    # we'll compute a running count of prior wins for each pair of fighters.
    # canonical_key is a tuple of the two names sorted lexicographically so that
    # events where the fighters swap red/blue sides are still grouped together.
    history: dict[tuple[str, str], dict[str, int]] = {}
    prev_wins = []
    for idx, row in df.iterrows():
        ra = row['r_fighter']
        ba = row['b_fighter']
        key = tuple(sorted([ra, ba]))
        pair_hist = history.setdefault(key, {})
        # how many times has each fighter beaten the other before this row?
        ra_wins = pair_hist.get(ra, 0)
        ba_wins = pair_hist.get(ba, 0)
        # store difference (red minus blue)
        prev_wins.append(ra_wins - ba_wins)
        # update history with this fight's outcome
        winner = row['winner']
        if winner == 'Red':
            pair_hist[ra] = ra_wins + 1
        elif winner == 'Blue':
            pair_hist[ba] = ba_wins + 1
        # draws/other values are ignored
    df = df.copy()
    df['prev_wins_against_opp'] = prev_wins
    base_cols = ['r_fighter', 'b_fighter', 'winner', 'prev_wins_against_opp']
    print(f"computed prev_wins for {len(df)} fights")
    return df[base_cols]


if __name__ == '__main__':
    df = extract()
    outname = os.path.splitext(os.path.basename(__file__))[0] + '.tsv'
    outpath = os.path.join(os.path.dirname(__file__), outname)
    df.to_csv(outpath, sep='\t', index=False)
    print(f"Wrote {outpath}")

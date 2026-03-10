import argparse
import pandas as pd, torch, os
from model.neural_network import UFCPredictor, FeatureDataset

FEATURE_FILES=[
    'extract/age/age.tsv',
    'extract/age_delta/age_delta.tsv',
    'extract/height_delta/height_delta.tsv',
    'extract/reach_delta/reach_delta.tsv',
    'extract/weight_delta/weight_delta.tsv',
    'extract/takedown_defense_delta/takedown_defense_delta.tsv',
    'extract/takedown_avg_delta/takedown_avg_delta.tsv',
    'extract/takedown_accuracy_delta/takedown_accuracy_delta.tsv',
    'extract/striking_advantage/striking_advantage.tsv',
    'extract/prev_wins_against_opp/prev_wins_against_opp.tsv',
    'extract/grappling_score/grappling_score.tsv',
    'extract/sig_strike_absorbed_delta/sig_strike_absorbed_delta.tsv',
    'extract/sig_strike_accuracy_delta/sig_strike_accuracy_delta.tsv',
    'extract/sig_strike_defense_delta/sig_strike_defense_delta.tsv',
    'extract/sig_strikes_pm_delta/sig_strikes_pm_delta.tsv',
    'extract/submission_avg_delta/submission_avg_delta.tsv',
]

# helper to load merged data once, with simple on-disk cache
CACHE_FILE = 'merged_features.pkl'

def load_merged():
    """Return the merged DataFrame of all feature files.

    The first time this is called we read and merge the TSVs; the
    result is pickled to disk. Subsequent calls will reuse the cached
    DataFrame as long as none of the source files have been modified
    since the pickle was written. This avoids paying the cost of
    merging on every invocation of the prediction script.
    """
    # check cache freshness
    if os.path.exists(CACHE_FILE):
        cache_mtime = os.path.getmtime(CACHE_FILE)
        stale = False
        for f in FEATURE_FILES:
            try:
                if os.path.getmtime(f) > cache_mtime:
                    stale = True
                    break
            except OSError:
                stale = True
                break
        if not stale:
            try:
                return pd.read_pickle(CACHE_FILE)
            except Exception:
                pass  # fall through to rebuild
    # rebuild
    dfs = [pd.read_csv(f, sep='\t') for f in FEATURE_FILES]
    df = dfs[0]
    for other in dfs[1:]:
        on=['r_fighter','b_fighter','winner']
        if 'weight_class' in df.columns and 'weight_class' in other.columns:
            on.append('weight_class')
        new_cols=[c for c in other.columns if c not in df.columns or c in on]
        df = df.merge(other[new_cols], on=on, how='inner')
    if 'weight_diff' in df.columns and 'weight_delta' not in df.columns:
        df = df.rename(columns={'weight_diff':'weight_delta'})
    try:
        df.to_pickle(CACHE_FILE)
    except Exception:
        pass
    return df

# compute average stats per fighter for raw and diff columns
def compute_player_averages(df):
    # build a long table where each contest contributes a row for the
    # red fighter (diffs as-is) and a row for the blue fighter (diffs
    # with sign flipped).
    base={'r_fighter','b_fighter','winner','weight_class'}
    diff_cols=[c for c in df.columns if 'diff' in c and c not in base]
    raw_cols=[c for c in df.columns if c not in base and c not in diff_cols and not c.startswith('r_') and not c.startswith('b_')]

    # red side
    red = df[['r_fighter'] + diff_cols + raw_cols + (['weight_class'] if 'weight_class' in df.columns else [])].copy()
    red = red.rename(columns={'r_fighter':'fighter'})

    # blue side: flip diff sign
    blue = df[['b_fighter'] + diff_cols + raw_cols + (['weight_class'] if 'weight_class' in df.columns else [])].copy()
    blue = blue.rename(columns={'b_fighter':'fighter'})
    for col in diff_cols:
        blue[col] = -blue[col]

    allf = pd.concat([red, blue], ignore_index=True)
    # compute mean of each numeric column grouped by fighter
    # only average numeric columns (weight_class is string, etc.)
    grouped = allf.groupby('fighter', dropna=False).mean(numeric_only=True)

    players = {}
    for fighter, row in grouped.iterrows():
        if pd.isna(fighter):
            continue
        stats = row.to_dict()
        # split into diff and raw
        avgdiff = {c:stats[c] for c in diff_cols if c in stats}
        avgraw = {c:stats[c] for c in raw_cols if c in stats}
        wc = None
        if 'weight_class' in stats and not pd.isna(stats['weight_class']):
            wc = stats['weight_class']
        players[fighter] = {'diff':avgdiff, 'raw':avgraw, 'weight_class':wc}
    return players

# given two fighter names and averages, make a feature row
def synthesize_row(f1,f2,players):
    if f1 not in players or f2 not in players:
        return None
    p1=players[f1]
    p2=players[f2]
    row={}
    # copy diff features as p1 - p2
    for col in set(p1['diff'].keys()) | set(p2['diff'].keys()):
        v1=p1['diff'].get(col,0.0)
        v2=p2['diff'].get(col,0.0)
        row[col]=v1-v2
    # for raw features compute difference as well
    for col in set(p1['raw'].keys()) | set(p2['raw'].keys()):
        v1=p1['raw'].get(col,0.0)
        v2=p2['raw'].get(col,0.0)
        row[col]=v1-v2
    # identifiers
    row['r_fighter']=f1
    row['b_fighter']=f2
    row['winner']='Red'  # dummy
    wc = p1.get('weight_class',None) or p2.get('weight_class',None)
    if isinstance(wc, (pd.Series,)):
        wc = wc.iloc[0] if len(wc) else None
    row['weight_class']=wc
    return pd.DataFrame([row])

# load ensembles similar to ensemble_predict
import re

def load_ensembles():
    ensembles={}
    pat=re.compile(r'^(.*)_e\d+\.pt$')
    for fname in os.listdir('model/checkpoints'):
        m=pat.match(fname)
        if m:
            wc=m.group(1)
            # checkpoint filenames use underscores instead of spaces
            wc = wc.replace('_',' ')
            ensembles.setdefault(wc,[]).append(os.path.join('model/checkpoints',fname))
    return ensembles


def build_match_index(df):
    """Return a dict mapping (fighter1,fighter2) -> (row, swapped).

    The ``row`` is the original pandas Series from the merged
    dataframe. ``swapped`` is True if the pair entry is the reverse of
    the stored row (i.e. fighter1 was blue)."""
    idx = {}
    for _, row in df.iterrows():
        r = row['r_fighter']
        b = row['b_fighter']
        idx[(r,b)] = (row, False)
        idx[(b,r)] = (row, True)
    return idx


def get_match_row(f1, f2, df, match_index):
    """Fast lookup that returns a single-row DataFrame or None.

    If the stored row has the fighters swapped relative to the request,
    the returned frame has its diff columns flipped to put ``f1`` on
    the red side."""

# predict winner row -> returns 'Red' or 'Blue'
def predict_from_row(df_row, ensembles, features):
    # extract the weight class from the row; it may come as a scalar, a
    # one-element Series, or even a numpy array.
    if isinstance(df_row, pd.DataFrame):
        wc = df_row.loc[df_row.index[0], 'weight_class']
    else:
        wc = df_row.get('weight_class')
    # unwrap containers
    if isinstance(wc, (pd.Series, list, tuple)):
        wc = wc.iloc[0] if isinstance(wc, pd.Series) else wc[0]
    try:
        import numpy as _np
        if isinstance(wc, _np.ndarray):
            wc = wc.item() if wc.size == 1 else wc[0]
    except ImportError:
        pass
    if wc is not None:
        wc = str(wc).strip()
    if wc not in ensembles:
        return None
    # normalize row using FeatureDataset
    ds=FeatureDataset(df_row, features)
    xs=ds.features
    logits_sum=torch.zeros_like(xs)
    count=0
    for mf in ensembles[wc]:
        m=UFCPredictor(input_dim=xs.shape[1], hidden1=64, hidden2=32)
        try:
            m.load_state_dict(torch.load(mf))
        except Exception:
            continue
        m.eval()
        with torch.no_grad():
            logits_sum += torch.sigmoid(m(xs))
        count+=1
    if count==0:
        return None
    avg=logits_sum/count
    return 'Red' if avg.item()>0.5 else 'Blue'


def compute_ensemble_accuracy(df, features, ensembles):
    """Evaluate the loaded ensembles on the merged dataframe.

    Returns a float between 0 and 1.
    """
    correct = 0
    total = 0
    for _, row in df.iterrows():
        wc = row.get('weight_class', None)
        if wc is None or wc not in ensembles:
            continue
        # craft a one-row df so FeatureDataset normalizes properly
        one = pd.DataFrame([row])
        ds = FeatureDataset(one, features)
        xs = ds.features
        logits_sum = torch.zeros_like(xs)
        loaded = 0
        for mf in ensembles[wc]:
            try:
                m = UFCPredictor(input_dim=xs.shape[1], hidden1=64, hidden2=32)
                m.load_state_dict(torch.load(mf))
            except Exception:
                continue
            m.eval()
            with torch.no_grad():
                logits_sum += torch.sigmoid(m(xs))
            loaded += 1
        if loaded == 0:
            continue
        avg = logits_sum / loaded
        pred = 'Red' if avg.item() > 0.5 else 'Blue'
        actual = row['winner']
        if pred == actual:
            correct += 1
        total += 1
    return correct / total if total else 0.0


def get_match_row(f1, f2, df, match_index):
    entry = match_index.get((f1, f2))
    if entry is None:
        return None
    row, swapped = entry
    row = row.to_frame().T.copy()
    if swapped:
        # flip as before
        row['r_fighter'], row['b_fighter'] = row['b_fighter'], row['r_fighter']
        if 'age_diff' in row.columns:
            row['age_diff'] = -row['age_diff']
        for c in row.columns:
            if 'diff' in c and c not in ['age_diff']:
                row[c] = -row[c]
    if 'weight_class' in row.columns:
        val = row.at[row.index[0], 'weight_class']
        if isinstance(val, pd.Series):
            row.at[row.index[0], 'weight_class'] = val.iloc[0]
    return row


def main():
    parser=argparse.ArgumentParser(description='Predict upcoming fight by fighter names')
    parser.add_argument('fighter1')
    parser.add_argument('fighter2')
    args=parser.parse_args()

    df=load_merged()
    players=compute_player_averages(df)
    ensembles=load_ensembles()

    # build auxiliary lookup structures
    match_index = build_match_index(df)

    # attempt to fetch existing contest; else synthesise
    row = get_match_row(args.fighter1, args.fighter2, df, match_index)
    if row is None:
        row = synthesize_row(args.fighter1, args.fighter2, players)
        if row is None:
            missing = [n for n in (args.fighter1, args.fighter2) if n not in players]
            print(f"fighter(s) {', '.join(missing)} not in dataset; cannot predict")
            return

    base={'r_fighter','b_fighter','winner','weight_class'}
    features=[c for c in row.columns if c not in base]

    # compute and display ensemble accuracy on merged data once
    acc = compute_ensemble_accuracy(df, features, ensembles)

    winner = predict_from_row(row, ensembles, features)
    if winner is None:
        print('no model available for weight class', row.get('weight_class'))
    else:
        print(f'predicted winner: {winner}')
        print(f'ensemble accuracy on training data: {acc:.2%}')

if __name__=='__main__':
    main()

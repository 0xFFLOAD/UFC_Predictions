#!/usr/bin/env python3
"""Print effective input weights for a trained checkpoint.

We approximate the contribution of each input feature by linearising the
network at zero (i.e. assume tanh' ~ 1).  For the architecture used in the
repository this amounts to computing

    w_eff = w2 @ w1 @ w0

where w0, w1, w2 are the weight matrices of the first, second and output
layers respectively.  The script loads a checkpoint (default: first found)
and prints a sorted list of features with their corresponding weight.

Usage:
    python show_feature_weights.py [checkpoint.pt]

The feature ordering is determined by merging the extracts exactly as in
training; the same merge code is reused here to ensure consistency.
"""

import sys
import glob
import pandas as pd
import torch

# merge as train.py/ensemble_predict
feature_files = [
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

def merge_features():
    dfs=[pd.read_csv(f,sep='\t') for f in feature_files]
    df=dfs[0]
    for other in dfs[1:]:
        on=['r_fighter','b_fighter','winner']
        if 'weight_class' in df.columns and 'weight_class' in other.columns:
            on.append('weight_class')
        new_cols=[c for c in other.columns if c not in df.columns or c in on]
        df=df.merge(other[new_cols], on=on, how='inner')
    if 'weight_diff' in df.columns and 'weight_delta' not in df.columns:
        df=df.rename(columns={'weight_diff':'weight_delta'})
    base={'r_fighter','b_fighter','winner','weight_class'}
    features=[c for c in df.columns if c not in base]
    return features

if __name__=='__main__':
    if len(sys.argv)>1:
        ckpt=sys.argv[1]
    else:
        candidates=glob.glob('model/checkpoints/*_e*.pt')
        if not candidates:
            print('no checkpoint found')
            sys.exit(1)
        ckpt=candidates[0]
    sd=torch.load(ckpt)
    # extract weight matrices
    w0=sd['net.0.weight'].cpu()
    w1=sd.get('net.2.weight',None)
    w2=sd.get('net.4.weight',None)
    if w1 is None or w2 is None:
        print('unexpected architecture')
        sys.exit(1)
    # compute effective weights: w_eff = w2 @ w1 @ w0
    # shapes: w0: (h1,inp)  w1:(h2,h1)  w2:(1,h2)
    w_eff = w2.matmul(w1.matmul(w0))
    w_eff = w_eff.squeeze(0).numpy()  # length = input dim
    features = merge_features()
    if len(features)!=len(w_eff):
        print('feature count mismatch', len(features), len(w_eff))
    # make sure we only pair as many features as weights
    if len(features) > len(w_eff):
        features = features[:len(w_eff)]
    # print sorted by magnitude
    pairs=list(zip(features, w_eff))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    for f,w in pairs:
        print(f'{f:30s} {w:.6e}')
    print('\n(using checkpoint', ckpt, ')')

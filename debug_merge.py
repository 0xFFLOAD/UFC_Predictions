#!/usr/bin/env python3
import pandas as pd

feature_files=[
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

dfs = [pd.read_csv(f, sep='\t') for f in feature_files]
df = dfs[0]
for other in dfs[1:]:
    on=['r_fighter','b_fighter','winner']
    if 'weight_class' in df.columns and 'weight_class' in other.columns:
        on.append('weight_class')
    new_cols=[c for c in other.columns if c not in df.columns or c in on]
    df = df.merge(other[new_cols], on=on, how='inner')
if 'weight_diff' in df.columns and 'weight_delta' not in df.columns:
    df = df.rename(columns={'weight_diff':'weight_delta'})

print('merged columns', len(df.columns))
print(df.columns.tolist())
print('sample row')
print(df.iloc[0])
print('dtype counts')
print(df.dtypes.value_counts())


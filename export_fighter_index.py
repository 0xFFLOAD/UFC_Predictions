#!/usr/bin/env python3
"""Build a per-fighter index and dump it as JSON.

The structure is simple: each fighter name maps to an object containing

    {
        "mean_stats": { feature: average_value, ... },
        "fights": [
            {"opponent": "Name", "won": true, "weight_class": "..."},
            ...
        ]
    }

The "mean_stats" values are computed from the same features that the
training pipeline uses; for opponents that were "Blue" in a given row
we flip any delta/diff-type feature so that the value is always from the
fighter's perspective.

Usage:
    python export_fighter_index.py > fighter_index.json

You can then load the resulting JSON into C, Python, etc.  A very small
C-friendly example is given in the comments below.
"""

import json
import pandas as pd
import sys


# copy of the merge logic from ensemble_predict.py / train.py
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
    # add any new feature files here
]

dfs = [pd.read_csv(f, sep='\t') for f in feature_files]
df = dfs[0]
for other in dfs[1:]:
    on = ['r_fighter', 'b_fighter', 'winner']
    if 'weight_class' in df.columns and 'weight_class' in other.columns:
        on.append('weight_class')
    new_cols = [c for c in other.columns if c not in df.columns or c in on]
    df = df.merge(other[new_cols], on=on, how='inner')

# aliasing
if 'weight_diff' in df.columns and 'weight_delta' not in df.columns:
    df = df.rename(columns={'weight_diff': 'weight_delta'})

base = {'r_fighter', 'b_fighter', 'winner', 'weight_class'}
features = [c for c in df.columns if c not in base]
clean = df.dropna(subset=features).reset_index(drop=True)

# build index
index = {}
for _, row in clean.iterrows():
    rname = row['r_fighter']
    bname = row['b_fighter']
    wc = row.get('weight_class', '')
    winner = row['winner']

    for fighter, opponent, is_red in [(rname, bname, True), (bname, rname, False)]:
        entry = index.setdefault(fighter, {
            'fights': [],
            'stats_acc': {f: 0.0 for f in features},
            'count': 0,
        })

        entry['fights'].append({
            'opponent': opponent,
            'won': winner == ('Red' if is_red else 'Blue'),
            'weight_class': wc,
        })

        # accumulate stats from perspective of this fighter; flip diffs
        for f in features:
            val = row[f]
            if isinstance(val, (int, float)):
                if (not is_red) and ('diff' in f or f.endswith('_delta')):
                    val = -val
                entry['stats_acc'][f] += val
        entry['count'] += 1

# convert accumulators to means
for fighter, data in index.items():
    cnt = data.pop('count')
    acc = data.pop('stats_acc')
    if cnt > 0:
        data['mean_stats'] = {f: acc[f] / cnt for f in features}
    else:
        data['mean_stats'] = {}

json.dump(index, fp=sys.stdout, indent=2, sort_keys=True)

# --- C example ---------------------------------------------------------
#
# The JSON file produced above can be parsed in C using a library such as
# cJSON, jsmn, or json-c.  An in-memory representation might look like:
#
#    typedef struct Fight {
#        const char *opponent;
#        int won;
#        const char *weight_class;
#        struct Fight *next;
#    } Fight;
#
#    typedef struct Fighter {
#        const char *name;
#        /* could also use a fixed struct with named fields instead of array */
#        double *mean_stats;      /* length = n_features */
#        const char **feature_names;
#        Fight *fights;
#        struct Fighter *next;    /* linked list for hash-bucket or simple list */
#    } Fighter;
#
# One could generate a .c/.h pair directly from the JSON data by writing a
# short Python script that emits static declarations, for example:
#
#    static Fighter fighters[] = {
#        {"Conor McGregor", ... , .fights = (Fight[]){ {"Nate Diaz",1,"Welterweight",...}, ... }},
#        ...
#    };
#
# or build a hash-table at runtime.  The pointer fields in the struct allow
# you to traverse each fighter's previous opponents without expensive searches.
#
# The JSON dump makes it easy to preprocess the data in Python and then feed
# it to C or any other language you choose.
#

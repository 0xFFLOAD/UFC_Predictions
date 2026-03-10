#!/usr/bin/env python3
"""Generate a self-contained C data file containing the fighter index.

This is an alternative to exporting JSON; it produces two files:

    fighter_index_data.h  -- forward declarations
    fighter_index_data.c  -- static arrays of Fighter and Fight

The generated C source can be compiled without any runtime JSON library.

Usage::

    python export_fighter_index_cfile.py > /dev/null

(The script writes the two files directly.)

The output uses the same feature ordering as ``model_weights.h`` so that the
``mean_stats`` arrays line up with the model input vector.
"""

import json, re, sys
import pandas as pd

# reuse merging logic from earlier
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
]

def merge_all():
    dfs = [pd.read_csv(f, sep='\t') for f in feature_files]
    df = dfs[0]
    for other in dfs[1:]:
        on = ['r_fighter', 'b_fighter', 'winner']
        if 'weight_class' in df.columns and 'weight_class' in other.columns:
            on.append('weight_class')
        new_cols = [c for c in other.columns if c not in df.columns or c in on]
        df = df.merge(other[new_cols], on=on, how='inner')
    if 'weight_diff' in df.columns and 'weight_delta' not in df.columns:
        df = df.rename(columns={'weight_diff': 'weight_delta'})
    return df

# load weights header to capture feature names order
import os
if not os.path.exists('model_weights.h'):
    sys.exit('run export_model_c.py first to generate model_weights.h')

# parse feature names out of header
feature_names = []
with open('model_weights.h') as f:
    for line in f:
        m = re.match(r"\s*\"(.*?)\",", line)
        if m:
            feature_names.append(m.group(1))

# build fighter averages same as before

df = merge_all()
base = {'r_fighter','b_fighter','winner','weight_class'}
features = [c for c in df.columns if c not in base]
clean = df.dropna(subset=features).reset_index(drop=True)

# compute per-fighter means with diff flipping
players = {}

diff_cols = [c for c in clean.columns if 'diff' in c and c not in base]
raw_cols = [c for c in clean.columns if c not in base and c not in diff_cols and not c.startswith('r_') and not c.startswith('b_')]

for _, row in clean.iterrows():
    r = row['r_fighter']; b = row['b_fighter']; wc = row.get('weight_class', None)
    winner = row['winner']
    for fighter, opp, is_red in [(r, b, True), (b, r, False)]:
        if fighter not in players:
            players[fighter] = {'diff':{}, 'raw':{}, 'weight_class':wc, 'count':0}
        ent = players[fighter]
        ent['fights'] = ent.get('fights', [])
        ent['fights'].append((opp, winner == ('Red' if is_red else 'Blue'), wc))
        # accumulate stats
        for c in diff_cols:
            v = row[c]
            if not is_red and pd.notna(v):
                v = -v
            ent['diff'][c] = ent['diff'].get(c, 0.0) + (v if pd.notna(v) else 0.0)
        for c in raw_cols:
            v = row[c]
            ent['raw'][c] = ent['raw'].get(c, 0.0) + (v if pd.notna(v) else 0.0)
        ent['count'] += 1

# convert to means
for fighter, ent in players.items():
    cnt = ent['count']
    if cnt > 0:
        for c in list(ent['diff'].keys()):
            ent['diff'][c] /= cnt
        for c in list(ent['raw'].keys()):
            ent['raw'][c] /= cnt

# now create C source

def sanitize(name):
    return re.sub(r'[^A-Za-z0-9]', '_', name)

# open files for writing
hfile = open('fighter_index_data.h','w')
cfile = open('fighter_index_data.c','w')

hfile.write('#ifndef FIGHTER_INDEX_DATA_H\n')
hfile.write('#define FIGHTER_INDEX_DATA_H\n\n')
hfile.write('#include "fighter_index.h"\n')
hfile.write('#include "model_weights.h"\n\n')

# forward declarations
hfile.write('extern const size_t n_fighters;\n')
hfile.write('extern Fighter fighters[];\n')

hfile.write('\n#endif\n')

# source file
cfile.write('#include "fighter_index_data.h"\n')
cfile.write('#include "model_weights.h"\n')
cfile.write('#include <stddef.h>\n\n')

# generate fights arrays and fighter structs
cfile.write('/* automatically generated fighter data */\n\n')

for idx,(fighter, ent) in enumerate(players.items()):
    fname = sanitize(fighter)
    # fights
    fights = ent.get('fights', [])
    cfile.write(f'static Fight fights_{fname}[] = {{\n')
    for opp, won, wc in fights:
        wc_str = wc if wc is not None else "\"\""
        cfile.write(f'    {{ "{opp}", {1 if won else 0}, "{wc if wc is not None else ""}", NULL }},\n')
    cfile.write('};\n\n')

# feature names array is global from model_weights.h; we'll assume same order

# for each fighter, output mean_stats array
for idx,(fighter, ent) in enumerate(players.items()):
    fname = sanitize(fighter)
    cfile.write(f'static double stats_{fname}[MODEL_INPUT_DIM] = {{')
    # compute v1-v2 earlier? no, we just store fighter means; prediction will subtract
    # Need to match order of feature_names
    for i,fn in enumerate(feature_names):
        # value from diff or raw or 0
        v = ent['diff'].get(fn)
        if v is None:
            v = ent['raw'].get(fn, 0.0)
        cfile.write(f'{v:.6e}')
        if i != len(feature_names)-1:
            cfile.write(', ')
    cfile.write('};\n')

# now create Fighter array with pointers
cfile.write('\nFighter fighters[] = {\n')
for idx,(fighter, ent) in enumerate(players.items()):
    fname = sanitize(fighter)
    cfile.write('    {\n')
    cfile.write(f'        "{fighter}",\n')
    cfile.write(f'        stats_{fname},\n')
    cfile.write('        (char**)model_feature_names,  /* reuse global names */\n')
    cfile.write('        MODEL_INPUT_DIM,\n')
    cfile.write(f'        fights_{fname},\n')
    cfile.write('        NULL\n')
    cfile.write('    },\n')
cfile.write('};\n\n')

cfile.write('const size_t n_fighters = sizeof(fighters)/sizeof(fighters[0]);\n')

hfile.close()
cfile.close()
print('generated fighter_index_data.h/c with', len(players), 'fighters')

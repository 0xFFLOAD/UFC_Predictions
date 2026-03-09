import pandas as pd, torch, os
from model.neural_network import UFCPredictor, FeatureDataset

# list of feature files weve been using
files=[
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
# merge them as train.py does

dfs=[pd.read_csv(f,sep='\t') for f in files]
df=dfs[0]
for other in dfs[1:]:
    on=['r_fighter','b_fighter','winner']
    if 'weight_class' in df.columns and 'weight_class' in other.columns:
        on.append('weight_class')
    new_cols=[c for c in other.columns if c not in df.columns or c in on]
    df=df.merge(other[new_cols], on=on, how='inner')

# rename weight_diff alias if needed
if 'weight_diff' in df.columns and 'weight_delta' not in df.columns:
    df=df.rename(columns={'weight_diff':'weight_delta'})

# features list (hard-coded for clarity)
features=['age_diff','height_diff','reach_diff','weight_delta',
          'td_def_total_diff','td_avg_diff','td_acc_total_diff',
          'striking_advantage','prev_wins_against_opp','grappling_score']

# drop rows missing any feature
clean=df.dropna(subset=features).reset_index(drop=True)
print(f'merged {len(df)} rows, {len(clean)} after dropping missing features')

# load per-class ensembles
import re

ensembles={}  # weight_class -> [model,model,model]
pattern = re.compile(r'^(.*)_e\d+\.pt$')
for fname in os.listdir('model/checkpoints'):
    m = pattern.match(fname)
    if not m:
        continue
    wc = m.group(1)
    ensembles.setdefault(wc, []).append(os.path.join('model/checkpoints',fname))

print('found ensembles for classes:', list(ensembles.keys()))

# compute predictions
correct=0
total=0
for idx,row in clean.iterrows():
    wc=row.get('weight_class','')
    if wc not in ensembles:
        continue
    vals=[row[f] for f in features]
    xs=torch.tensor([vals],dtype=torch.float32)

    # need same normalization used during training: create dataset on-one-row
    ds=FeatureDataset(pd.DataFrame([row]), features)
    xnorm=ds.features
    logits_sum=torch.zeros((1,1))
    loaded = 0
    for model_path in ensembles[wc]:
        try:
            model=UFCPredictor(input_dim=len(features),hidden1=64,hidden2=32)
            model.load_state_dict(torch.load(model_path))
            print(f"    loaded checkpoint {model_path}")
        except Exception as e:
            print(f"    skipping incompatible checkpoint {model_path}: {e}")
            continue
        model.eval()
        with torch.no_grad():
            logits_sum += torch.sigmoid(model(xnorm))
        loaded += 1
    if loaded == 0:
        continue
    avg=logits_sum / loaded
    pred=(avg>0.5).float().item()
    actual = 1.0 if row['winner']=='Red' else 0.0
    if pred==actual:
        correct+=1
    total+=1

print(f'ensemble accuracy on merged dataset: {correct}/{total} = {correct/total:.2%}')

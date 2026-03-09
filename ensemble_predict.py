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

# compute feature list automatically (all non-identifier columns)
base = {'r_fighter','b_fighter','winner','weight_class'}
# after merging we may still have r_age/b_age and diff columns; include them
features = [c for c in df.columns if c not in base]
# drop rows missing any of the chosen features
clean = df.dropna(subset=features).reset_index(drop=True)
print(f'merged {len(df)} rows, {len(clean)} after dropping missing features')
print('using features:', features)

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
            sd = torch.load(model_path)
            # infer architecture from state dict shapes
            w0 = sd['net.0.weight']
            hidden1 = w0.shape[0]
            in_dim = w0.shape[1]
            w1 = sd.get('net.2.weight', None)
            if w1 is not None:
                hidden2 = w1.shape[0]
            else:
                hidden2 = 0
            # construct model accordingly
            model = UFCPredictor(input_dim=in_dim,
                                 hidden1=hidden1,
                                 hidden2=hidden2)
            model.load_state_dict(sd)
            print(f"    loaded checkpoint {model_path} (in={in_dim},h1={hidden1},h2={hidden2})")
        except Exception as e:
            print(f"    skipping incompatible checkpoint {model_path}: {e}")
            continue
        model.eval()
        # adjust xnorm if model expects fewer features than we have
        x_input = xnorm
        if x_input.shape[1] != in_dim:
            # attempt to drop grappling_score if it's causing mismatch
            if x_input.shape[1] - in_dim == 1 and 'grappling_score' in features:
                idx = features.index('grappling_score')
                print(f"    dropping grappling_score column for model {model_path}")
                x_input = torch.cat([x_input[:, :idx], x_input[:, idx+1:]], dim=1)
            else:
                print(f"    dim mismatch skipping {model_path} (have {x_input.shape[1]} expected {in_dim})")
                continue
        with torch.no_grad():
            logits_sum += torch.sigmoid(model(x_input))
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

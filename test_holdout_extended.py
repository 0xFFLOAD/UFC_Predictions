import pandas as pd, torch, random
from model.neural_network import UFCPredictor, train_model, FeatureDataset

# load all features available
files=['extract/age/age.tsv',
       'extract/age_delta/age_delta.tsv',
       'extract/height_delta/height_delta.tsv',
       'extract/reach_delta/reach_delta.tsv',
       'extract/weight_delta/weight_delta.tsv',
       'extract/takedown_defense_delta/takedown_defense_delta.tsv']

dfs=[pd.read_csv(f,sep='\t') for f in files]

df=dfs[0]
for other in dfs[1:]:
    on=['r_fighter','b_fighter','winner']
    if 'weight_class' in df.columns and 'weight_class' in other.columns:
        on.append('weight_class')
    df = df.merge(other, on=on, how='inner')

# sample 50 blind after merging
indices=random.sample(list(df.index), 50)
test=df.loc[indices]
train=df.drop(indices)

# choose all non-identifier columns as features
base={'r_fighter','b_fighter','winner','weight_class'}
features=[c for c in train.columns if c not in base]

# rename weight_diff alias if needed
if 'weight_diff' in train.columns and 'weight_delta' not in train.columns:
    train=train.rename(columns={'weight_diff':'weight_delta'})
    test=test.rename(columns={'weight_diff':'weight_delta'})
    features=[f if f!='weight_diff' else 'weight_delta' for f in features]

print(f"train {len(train)}, test {len(test)}, features {features}")

model=UFCPredictor(input_dim=len(features),hidden1=64,hidden2=32)
train_model(model, train, features, epochs=30, lr=0.001, batch_size=32)

model.eval()
with torch.no_grad():
    ds=FeatureDataset(test, features)
    xs=ds.features
    ys=ds.labels
    logits=model(xs)
    preds=(torch.sigmoid(logits)>0.5).float()
    correct=(preds==ys).float().sum().item()
    acc=correct/len(test)
print(f'Test accuracy: {acc:.2%} ({correct}/{len(test)})')

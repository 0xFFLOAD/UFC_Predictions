import pandas as pd, torch, random
from model.neural_network import UFCPredictor, train_model, FeatureDataset

# merge all features
def load_df(path):
    return pd.read_csv(path, sep='\t')
files=['extract/age/age.tsv',
       'extract/age_delta/age_delta.tsv',
       'extract/height_delta/height_delta.tsv',
       'extract/reach_delta/reach_delta.tsv',
       'extract/weight/weight.tsv']
dfs=[load_df(f) for f in files]
df=dfs[0]
for other in dfs[1:]:
    on=['r_fighter','b_fighter','winner']
    if 'weight_class' in df.columns and 'weight_class' in other.columns:
        on.append('weight_class')
    df = df.merge(other, on=on, how='inner')

# sample
indices=random.sample(list(df.index), 20)
test=df.loc[indices]
train=df.drop(indices)

# prepare features
features=['r_age','b_age','age_diff','height_diff','reach_diff','weight_diff']
if 'weight_diff' in train.columns:
    train=train.rename(columns={'weight_diff':'weight_delta'})
    test=test.rename(columns={'weight_diff':'weight_delta'})
features=[f if f!='weight_diff' else 'weight_delta' for f in features]

print('train size',len(train),'test size',len(test))

model=UFCPredictor(input_dim=len(features),hidden1=256,hidden2=128,hidden3=64)
train_model(model, train, features, epochs=50, lr=0.001, batch_size=32)

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

import torch, pandas as pd
from model.neural_network import UFCPredictor

df=pd.read_csv('extract/age/age.tsv',sep='\t')
other=pd.read_csv('extract/age_delta/age_delta.tsv',sep='\t')
on_cols=['r_fighter','b_fighter','winner']
if 'weight_class' in df.columns and 'weight_class' in other.columns:
    on_cols.append('weight_class')
new=[c for c in other.columns if c not in df.columns or c in on_cols]
df=df.merge(other[new],on=on_cols,how='inner')
feats=['r_age','b_age','age_diff']
xs=torch.tensor(df[feats].values.astype(float),dtype=torch.float32)
ys=torch.tensor((df['winner']=='Red').astype(float)).unsqueeze(1)
model=UFCPredictor(input_dim=len(feats),hidden1=512,hidden2=128,hidden3=64)
model.load_state_dict(torch.load('model/checkpoints/model_win.pt'))
model.eval()
with torch.no_grad():
    out=torch.sigmoid(model(xs))
pred=(out>0.5).float()
print((pred==ys).float().mean().item())

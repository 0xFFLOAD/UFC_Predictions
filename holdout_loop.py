import pandas as pd, torch, random
from model.neural_network import UFCPredictor, train_model, FeatureDataset

def make_df():
    # include every feature we've experimented with so far
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
        'extract/prev_wins_against_opp/prev_wins_against_opp.tsv'
    ]
    dfs=[pd.read_csv(f,sep='\t') for f in files]
    df=dfs[0]
    for other in dfs[1:]:
        on=['r_fighter','b_fighter','winner']
        if 'weight_class' in df.columns and 'weight_class' in other.columns:
            on.append('weight_class')
        df = df.merge(other, on=on, how='inner')
    return df

runs=5
accs=[]
for i in range(runs):
    df=make_df()
    indices=random.sample(list(df.index),50)
    test=df.loc[indices]
    train=df.drop(indices)
    base={'r_fighter','b_fighter','winner','weight_class'}
    features=[c for c in train.columns if c not in base]
    if 'weight_diff' in train.columns and 'weight_delta' not in train.columns:
        train=train.rename(columns={'weight_diff':'weight_delta'})
        test=test.rename(columns={'weight_diff':'weight_delta'})
        features=[f if f!='weight_diff' else 'weight_delta' for f in features]
    model=UFCPredictor(input_dim=len(features),hidden1=64,hidden2=32)
    train_model(model, train, features, epochs=3, lr=0.001, batch_size=32)
    model.eval()
    with torch.no_grad():
        ds=FeatureDataset(test, features)
        logits=model(ds.features)
        preds=(torch.sigmoid(logits)>0.5).float()
        corr=(preds==ds.labels).float().sum().item()
    acc=corr/len(test)
    accs.append(acc)
    print(f'run {i+1}/{runs} acc={acc:.2%}')

avg=sum(accs)/len(accs)
print(f'average accuracy over {runs} runs: {avg:.2%}')

import pandas as pd, torch, random, itertools
from model.neural_network import UFCPredictor, train_model, FeatureDataset

# load all feature files
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
]

def make_df():
    dfs=[pd.read_csv(f,sep='\t') for f in files]
    df=dfs[0]
    for other in dfs[1:]:
        on=['r_fighter','b_fighter','winner']
        if 'weight_class' in df.columns and 'weight_class' in other.columns:
            on.append('weight_class')
        df = df.merge(other, on=on, how='inner')
    return df

full = make_df()
classes = sorted(full['weight_class'].dropna().unique())
print('weight classes found:', classes)

# configuration grid
hidden1_vals=[32,64,128]
hidden2_vals=[16,32,64]
epochs=20
lr=0.001

results = []
for wc in classes:
    df = full[full['weight_class']==wc].reset_index(drop=True)
    if len(df) < 200:
        print(f'skipping class {wc} (<200 examples)')
        continue
    print(f'processing class {wc}, {len(df)} rows')
    # sample held-out set of 200 (larger folds give more stable estimates)
    holdout_size=200
    indices=random.sample(list(df.index),holdout_size)
    test=df.loc[indices]
    train=df.drop(indices)
    base={'r_fighter','b_fighter','winner','weight_class'}
    features=[c for c in train.columns if c not in base]
    if 'weight_diff' in train.columns and 'weight_delta' not in train.columns:
        train=train.rename(columns={'weight_diff':'weight_delta'})
        test=test.rename(columns={'weight_diff':'weight_delta'})
        features=[f if f!='weight_diff' else 'weight_delta' for f in features]

    best_acc=0.0
    best_cfg=None
    # grid search for best hyperparameters
    for h1,h2 in itertools.product(hidden1_vals,hidden2_vals):
        try:
            model=UFCPredictor(input_dim=len(features),hidden1=h1,hidden2=h2)
            train_model(model, train, features, epochs=epochs, lr=lr, batch_size=32)
        except Exception as e:
            print(f'    error training cfg h1={h1} h2={h2}: {e}')
            continue
        model.eval()
        with torch.no_grad():
            ds=FeatureDataset(test, features)
            logits=model(ds.features)
            preds=(torch.sigmoid(logits)>0.5).float()
            corr=(preds==ds.labels).float().sum().item()
        acc=corr/len(test)
        print(f'  cfg h1={h1} h2={h2} -> acc={acc:.2%}')
        if acc>best_acc:
            best_acc=acc
            best_cfg=(h1,h2)
    results.append((wc,best_acc,best_cfg))

    # now train and save an ensemble of best models
    if best_cfg is not None:
        h1,h2 = best_cfg
        for i in range(3):
            torch.manual_seed(i)
            ensemble_model = UFCPredictor(input_dim=len(features), hidden1=h1, hidden2=h2)
            train_model(ensemble_model, train, features, epochs=epochs, lr=lr, batch_size=32)
            outpath = f"model/checkpoints/{wc.replace(' ','_')}_e{i}.pt"
            torch.save(ensemble_model.state_dict(), outpath)
            print(f"    saved ensemble model {outpath}")
        # evaluate ensemble on the holdout
        with torch.no_grad():
            ds=FeatureDataset(test, features)
            xs=ds.features
            logits_sum = torch.zeros_like(xs[:,0:1])
            for i in range(3):
                m = UFCPredictor(input_dim=len(features), hidden1=h1, hidden2=h2)
                m.load_state_dict(torch.load(f"model/checkpoints/{wc.replace(' ','_')}_e{i}.pt"))
                m.eval()
                logits_sum += torch.sigmoid(m(xs))
            preds = (logits_sum/3.0 > 0.5).float()
            corr = (preds==ds.labels).float().sum().item()
            ens_acc = corr/len(test)
            print(f"    ensemble accuracy on holdout: {ens_acc:.2%}")
    # continue next class even if some configs fail


print('\nsummary per-class best:')
for wc,acc,cfg in results:
    print(f'{wc}: {acc:.2%} (h1={cfg[0]} h2={cfg[1]})')

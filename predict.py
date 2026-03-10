import argparse
import pandas as pd, torch, os
from model.neural_network import UFCPredictor, FeatureDataset

FEATURE_FILES=[
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

# helper to load merged data once
def load_merged():
    dfs=[pd.read_csv(f,sep='\t') for f in FEATURE_FILES]
    df=dfs[0]
    for other in dfs[1:]:
        on=['r_fighter','b_fighter','winner']
        if 'weight_class' in df.columns and 'weight_class' in other.columns:
            on.append('weight_class')
        new_cols=[c for c in other.columns if c not in df.columns or c in on]
        df = df.merge(other[new_cols], on=on, how='inner')
    # alias weight_diff->weight_delta if necessary
    if 'weight_diff' in df.columns and 'weight_delta' not in df.columns:
        df = df.rename(columns={'weight_diff':'weight_delta'})
    return df

# compute average stats per fighter for raw and diff columns
def compute_player_averages(df):
    base={'r_fighter','b_fighter','winner','weight_class'}
    diff_cols=[c for c in df.columns if 'diff' in c and c not in base]
    raw_cols=[c for c in df.columns if c not in base and c not in diff_cols and not c.startswith('r_') and not c.startswith('b_')]
    players={}
    for name in pd.unique(df[['r_fighter','b_fighter']].values.ravel()):
        rows_r=df[df['r_fighter']==name]
        rows_b=df[df['b_fighter']==name]
        if rows_r.empty and rows_b.empty:
            continue
        avgdiff={}
        for col in diff_cols:
            vals=[]
            # when fighter is red, col is advantage of red over blue
            vals.extend(rows_r[col].dropna().tolist())
            # when fighter is blue, invert sign
            vals.extend([-v for v in rows_b[col].dropna().tolist()])
            if vals:
                avgdiff[col]=sum(vals)/len(vals)
        avgraw={}
        for col in raw_cols:
            vals=[]
            if col in rows_r.columns:
                vals.extend(rows_r[col].dropna().tolist())
            if col in rows_b.columns:
                vals.extend(rows_b[col].dropna().tolist())
            if vals:
                avgraw[col]=sum(vals)/len(vals)
        # majority weight class
        wclasses=pd.concat([rows_r['weight_class'], rows_b['weight_class']]).dropna()
        wc=wclasses.mode().iloc[0] if not wclasses.empty else None
        players[name]={'diff':avgdiff,'raw':avgraw,'weight_class':wc}
    return players

# given two fighter names and averages, make a feature row
def synthesize_row(f1,f2,players):
    if f1 not in players or f2 not in players:
        return None
    p1=players[f1]
    p2=players[f2]
    row={}
    # copy diff features as p1 - p2
    for col in set(p1['diff'].keys()) | set(p2['diff'].keys()):
        v1=p1['diff'].get(col,0.0)
        v2=p2['diff'].get(col,0.0)
        row[col]=v1-v2
    # for raw features compute difference as well
    for col in set(p1['raw'].keys()) | set(p2['raw'].keys()):
        v1=p1['raw'].get(col,0.0)
        v2=p2['raw'].get(col,0.0)
        row[col]=v1-v2
    # identifiers
    row['r_fighter']=f1
    row['b_fighter']=f2
    row['winner']='Red'  # dummy
    row['weight_class']=p1.get('weight_class',None) or p2.get('weight_class',None)
    return pd.DataFrame([row])

# load ensembles similar to ensemble_predict
import re

def load_ensembles():
    ensembles={}
    pat=re.compile(r'^(.*)_e\d+\.pt$')
    for fname in os.listdir('model/checkpoints'):
        m=pat.match(fname)
        if m:
            wc=m.group(1)
            ensembles.setdefault(wc,[]).append(os.path.join('model/checkpoints',fname))
    return ensembles

# predict winner row -> returns 'Red' or 'Blue'
def predict_from_row(df_row, ensembles, features):
    wc=df_row['weight_class']
    if wc not in ensembles:
        return None
    # normalize row using FeatureDataset
    ds=FeatureDataset(df_row, features)
    xs=ds.features
    logits_sum=torch.zeros_like(xs)
    count=0
    for mf in ensembles[wc]:
        m=UFCPredictor(input_dim=xs.shape[1], hidden1=64, hidden2=32)
        try:
            m.load_state_dict(torch.load(mf))
        except Exception:
            continue
        m.eval()
        with torch.no_grad():
            logits_sum += torch.sigmoid(m(xs))
        count+=1
    if count==0:
        return None
    avg=logits_sum/count
    return 'Red' if avg.item()>0.5 else 'Blue'


def main():
    parser=argparse.ArgumentParser(description='Predict upcoming fight by fighter names')
    parser.add_argument('fighter1')
    parser.add_argument('fighter2')
    args=parser.parse_args()

    df=load_merged()
    players=compute_player_averages(df)
    ensembles=load_ensembles()

    # try to find exact matchup
    match = df[(df['r_fighter']==args.fighter1) & (df['b_fighter']==args.fighter2)]
    if len(match)==0:
        match = df[(df['r_fighter']==args.fighter2) & (df['b_fighter']==args.fighter1)]
        swapped = True if len(match)>0 else False
    else:
        swapped = False
    if len(match)>0:
        row=match.iloc[[0]].copy()
        if swapped:
            # flip columns so fighter1 is red
            row['r_fighter'],row['b_fighter'] = row['b_fighter'],row['r_fighter']
            if 'age_diff' in row.columns:
                row['age_diff'] = -row['age_diff']
            # for each diff column swap sign
            for c in row.columns:
                if 'diff' in c and c not in ['age_diff']:
                    row[c] = -row[c]
    else:
        row=synthesize_row(args.fighter1,args.fighter2,players)
        if row is None:
            print('one or both fighters unknown to dataset; cannot predict')
            return
    base={'r_fighter','b_fighter','winner','weight_class'}
    features=[c for c in row.columns if c not in base]
    winner = predict_from_row(row, ensembles, features)
    if winner is None:
        print('no model available for weight class', row.get('weight_class'))
    else:
        print(f'predicted winner: {winner}')

if __name__=='__main__':
    main()

#!/usr/bin/env python3
import pandas as pd
from predict import load_merged, compute_player_averages, synthesize_row, load_ensembles, predict_from_row
import torch
from model.neural_network import UFCPredictor, FeatureDataset

def explain(f1, f2):
    df = load_merged()
    players = compute_player_averages(df)
    row = synthesize_row(f1,f2,players)
    if row is None:
        print('could not synthesize for', f1, f2)
        return
    features=[c for c in row.columns if c not in {'r_fighter','b_fighter','winner','weight_class'}]
    print('feature differences for', f1, 'minus', f2)
    for f in features:
        print(' ', f, row.at[0,f])
    ensembles=load_ensembles()
    wc=row.iloc[0]['weight_class']
    print('weight class', wc)
    winner = predict_from_row(row, ensembles, features)
    print('predicted', winner)
    ds=FeatureDataset(row, features)
    xs=ds.features
    logits_sum=torch.zeros_like(xs)
    cnt=0
    for mf in ensembles.get(wc, []):
        m=UFCPredictor(input_dim=xs.shape[1], hidden1=64, hidden2=32)
        m.load_state_dict(torch.load(mf))
        m.eval()
        with torch.no_grad():
            logits_sum += torch.sigmoid(m(xs))
        cnt+=1
    if cnt:
        prob=(logits_sum/cnt).item()
        print('probability red wins:', prob)

if __name__=='__main__':
    import sys
    if len(sys.argv)!=3:
        print('usage: explain_prediction.py fighter1 fighter2')
    else:
        explain(sys.argv[1], sys.argv[2])

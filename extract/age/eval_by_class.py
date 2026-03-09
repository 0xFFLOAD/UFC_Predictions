"""Evaluate age-based models per weight class against held data.

Loads all checkpoints in ``model/checkpoints`` and for each corresponding
``age_<class>.tsv`` computes prediction accuracy (and loss optionally).
Outputs a summary table.

Usage::

    cd extract/age
    python eval_by_class.py

The script assumes features ``['r_age','b_age']`` by default but you can
provide ``--features`` to override.  It also accepts a ``--batch`` size
for evaluation batching.
"""

import argparse
import glob
import os
import pandas as pd

import torch
from model.neural_network import UFCPredictor

CHECKPOINT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              '..', 'model', 'checkpoints'))


def evaluate(model, df, feature_columns):
    # prepare data
    df = df.dropna(subset=feature_columns)
    xs = torch.tensor(df[feature_columns].values.astype(float), dtype=torch.float32)
    if 'winner' not in df.columns:
        raise ValueError('no winner column present')
    # ensure we reference column by name, not by index
    ys = torch.tensor((df.loc[:, 'winner'] == 'Red').astype(float).values).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        logits = model(xs)
        preds = torch.sigmoid(logits)
        preds_label = (preds > 0.5).float()
        correct = (preds_label == ys).float().sum().item()
    acc = correct / len(df) if len(df) > 0 else 0.0
    # compute loss
    lossfn = torch.nn.BCEWithLogitsLoss()
    loss_val = lossfn(logits, ys).item()
    return acc, loss_val


def main():
    parser = argparse.ArgumentParser(description="Evaluate per-class age models")
    parser.add_argument('--features', nargs='+', default=['r_age', 'b_age'],
                        help='Feature columns to use')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold on sigmoid output')
    args = parser.parse_args()

    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, '*.pt'))
    if not checkpoints:
        print('No checkpoints found; run training first')
        return

    results = []
    for chk in checkpoints:
        cls = os.path.splitext(os.path.basename(chk))[0]
        # find tsv with matching class
        tsv = os.path.join(os.path.dirname(__file__), f'age_{cls}.tsv')
        if not os.path.exists(tsv):
            print(f'warning: no TSV for class {cls} (expected {tsv})')
            continue
        df = pd.read_csv(tsv, sep='\t')
        model = UFCPredictor(input_dim=len(args.features))
        model.load_state_dict(torch.load(chk))
        acc, loss = evaluate(model, df, args.features)
        results.append((cls, len(df), acc, loss))
        print(f'{cls}: {len(df)} samples, acc={acc:.3f}, loss={loss:.3f}')

    if results:
        print('\nSummary:')
        print('class\tsize\tacc\tloss')
        for cls, size, acc, loss in results:
            print(f'{cls}\t{size}\t{acc:.3f}\t{loss:.3f}')


if __name__ == '__main__':
    main()

"""Evaluate age-based models per weight class against held data.

Loads all checkpoints in ``model/checkpoints`` and for each corresponding
``age_<class>.tsv`` computes prediction accuracy (and loss optionally).
Outputs a summary table.

Usage::

    cd extract/age
    python eval_by_class.py

You may also pass ``--data`` with one or more feature TSVs; when multiple
sources are given they are merged per class just as in
``train_by_class.py``.  This lets you evaluate models on age, age-delta,
or any combination of features without creating a new file.

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


def evaluate(model, df, feature_columns, invert_flag=False):
    # prepare data
    df = df.dropna(subset=feature_columns)
    xs = torch.tensor(df[feature_columns].values.astype(float), dtype=torch.float32)
    if 'winner' not in df.columns:
        raise ValueError('no winner column present')
    ys = torch.tensor((df.loc[:, 'winner'] == 'Red').astype(float).values).unsqueeze(1)
    if invert_flag:
        ys = 1 - ys
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
    parser.add_argument('--data', nargs='+',
                        help='TSV files to merge for each class; if omitted will look for age_<class>.tsv in current directory')
    parser.add_argument('--invert', action='store_true',
                        help='Interpret checkpoints as loss models (labels inverted)')
    args = parser.parse_args()

    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, '*.pt'))
    if not checkpoints:
        print('No checkpoints found; run training first')
        return

    # determine source files
    if args.data:
        source_files = args.data
    else:
        pattern = os.path.join(os.path.dirname(__file__), 'age_*.tsv')
        source_files = [f for f in glob.glob(pattern) if os.path.basename(f) != 'age.tsv']

    # helper to load and filter by class
    def load_for_class(path, cls):
        df = pd.read_csv(path, sep='\t')
        if 'weight_class' in df.columns:
            df = df[df['weight_class'] == cls]
        return df

    results = []
    for chk in checkpoints:
        cls = os.path.splitext(os.path.basename(chk))[0]
        invert_flag = args.invert or chk.endswith('_loss.pt')
        # merge data for this class
        dfs = [load_for_class(path, cls) for path in source_files]
        if not dfs or dfs[0].empty:
            print(f'warning: no data for class {cls}')
            continue
        df = dfs[0]
        for other in dfs[1:]:
            on_cols = ['r_fighter', 'b_fighter', 'winner']
            if 'weight_class' in df.columns and 'weight_class' in other.columns:
                on_cols.append('weight_class')
            df = df.merge(other, on=on_cols, how='inner')
        model = UFCPredictor(input_dim=len(args.features))
        model.load_state_dict(torch.load(chk))
        acc, loss = evaluate(model, df, args.features, invert_flag=invert_flag)
        results.append((cls, len(df), acc, loss, invert_flag))
        tag = 'loss' if invert_flag else 'win'
        print(f'{cls} ({tag}): {len(df)} samples, acc={acc:.3f}, loss={loss:.3f}')

    if results:
        print('\nSummary:')
        print('class\tsize\tacc\tloss')
        for cls, size, acc, loss in results:
            print(f'{cls}\t{size}\t{acc:.3f}\t{loss:.3f}')


if __name__ == '__main__':
    main()

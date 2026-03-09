"""Convenience script for training the UFC predictor on TSV data.

Usage example::

    python model/train.py --features height_diff reach_diff ... \
        --data extract/age/age.tsv extract/height_delta/height_delta.tsv

The script concatenates any number of TSV files to build a training
DataFrame, then invokes the training routine defined in
``model/neural_network.py``.
"""

import argparse
import pandas as pd
from model.neural_network import UFCPredictor, train_model


def main():
    parser = argparse.ArgumentParser(description="Train UFC prediction model")
    parser.add_argument('--data', nargs='+', required=True,
                        help='TSV files containing feature columns and identifiers')
    parser.add_argument('--features', nargs='+', required=True,
                        help='List of feature column names to use as inputs')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs (ignored in search mode)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (ignored in search mode)')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size (ignored in search mode)')
    parser.add_argument('--search', action='store_true',
                        help='Perform grid search over hyperparameters')
    parser.add_argument('--lr-values', type=str,
                        help='Comma-separated list of lr values for grid search')
    parser.add_argument('--batch-values', type=str,
                        help='Comma-separated list of batch sizes for grid search')
    parser.add_argument('--epoch-values', type=str,
                        help='Comma-separated list of epoch counts for grid search')
    parser.add_argument('--auto-lr', action='store_true',
                        help='Run a learning-rate finder before training')
    parser.add_argument('--per-class', action='store_true',
                        help='Train separate models for each weight_class present in the data')
    args = parser.parse_args()

    # read all supplied files
    dfs = [pd.read_csv(f, sep='\t') for f in args.data]
    # if multiple files are provided, perform an inner join on the
    # fighter identifiers rather than simply stacking rows.  This lets
    # the caller supply distinct feature tables (e.g. age and age_delta)
    # and train on the combined set.
    if len(dfs) == 1:
        df = dfs[0]
    else:
        # start with the first frame and merge each subsequent one
        df = dfs[0]
        for other in dfs[1:]:
            # determine which columns to merge on; weight_class is
            # included when present so that per-class splits still work
            on_cols = ['r_fighter', 'b_fighter', 'winner']
            if 'weight_class' in df.columns and 'weight_class' in other.columns:
                on_cols.append('weight_class')
            df = df.merge(other, on=on_cols, how='inner')

    if args.search:
        # build value lists
        def parse_list(s, cast):
            return [cast(x) for x in s.split(',')] if s else []
        lrs = parse_list(args.lr_values, float)
        batches = parse_list(args.batch_values, int)
        epochs_list = parse_list(args.epoch_values, int)
        if not (lrs and batches and epochs_list):
            parser.error('Search mode requires --lr-values, --batch-values, and --epoch-values')

        best = None
        best_cfg = None
        for lr in lrs:
            for batch in batches:
                for epochs in epochs_list:
                    model = UFCPredictor(input_dim=len(args.features))
                    print(f'-> testing lr={lr}, batch={batch}, epochs={epochs}')
                    train_model(model, df, args.features,
                                epochs=epochs, lr=lr, batch_size=batch)
                    # after training we can compute loss on full set
                    ds = df.dropna(subset=args.features).reset_index(drop=True)
                    ds_model = model.eval()
                    import torch
                    xs = torch.tensor(ds[args.features].values.astype(float), dtype=torch.float32)
                    with torch.no_grad():
                        logits = ds_model(xs)
                        lossfn = torch.nn.BCEWithLogitsLoss()
                        ys = torch.tensor((ds['winner']=='Red').astype(float)).unsqueeze(1)
                        loss_val = lossfn(logits, ys).item()
                    print(f'    final loss: {loss_val:.4f}')
                    if best is None or loss_val < best:
                        best = loss_val
                        best_cfg = (lr, batch, epochs)
        print(f'BEST configuration: lr={best_cfg[0]}, batch={best_cfg[1]}, epochs={best_cfg[2]} -> loss={best:.4f}')
    else:
        if args.auto_lr:
            from model.neural_network import find_learning_rate
            print('running learning-rate finder...')
            lrs, losses, best_lr = find_learning_rate(df, args.features,
                                                      batch_size=args.batch)
            print(f'suggested lr = {best_lr:.6g}')
            args.lr = best_lr
        model = UFCPredictor(input_dim=len(args.features))
        train_model(model, df, args.features,
                    epochs=args.epochs, lr=args.lr, batch_size=args.batch)


if __name__ == '__main__':
    main()

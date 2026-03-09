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
    parser.add_argument('--features', nargs='+',
                        help='List of feature column names to use as inputs')
    parser.add_argument('--all-features', action='store_true',
                        help='Use every non-identifier column found in the merged data')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs (ignored in search mode)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (ignored in search mode)')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size (ignored in search mode)')
    parser.add_argument('--hidden1', type=int, default=64,
                        help='Size of first hidden layer')
    parser.add_argument('--hidden2', type=int, default=32,
                        help='Size of second hidden layer')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout probability (0 disables)')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='L2 regularization strength')
    parser.add_argument('--patience', type=int,
                        help='Early stopping patience (epochs)')
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
    parser.add_argument('--invert', action='store_true',
                        help='Flip labels (Red->0, Blue->1), i.e. predict loss instead of win')
    parser.add_argument('--double', action='store_true',
                        help='Train both win and loss models sequentially')
    parser.add_argument('--save', type=str,
                        help='Path prefix to save trained model(s); extension and suffix appended automatically')
    args = parser.parse_args()

    # read all supplied files
    dfs = [pd.read_csv(f, sep='\t') for f in args.data]
    if args.all_features:
        # compute feature list automatically as every column except
        # identifiers/winner/weight_class
        sample = dfs[0]
        base = {'r_fighter','b_fighter','winner','weight_class'}
        auto_feats = [c for c in sample.columns if c not in base]
        args.features = auto_feats
    if not args.features:
        parser.error('Must specify --features or --all-features')
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

    def do_train(label_invert: bool, prefix: str = None):
        # build model with provided architecture params
        model = UFCPredictor(input_dim=len(args.features),
                             hidden1=args.hidden1,
                             hidden2=args.hidden2,
                             dropout=args.dropout)
        # helper that either does a single training run or the search mode
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
                        print(f'-> testing lr={lr}, batch={batch}, epochs={epochs} invert={label_invert}')
                        train_model(model, df, args.features,
                                    epochs=epochs, lr=lr, batch_size=batch,
                                    invert=label_invert,
                                    weight_decay=args.weight_decay,
                                    patience=args.patience)
                        # after training we can compute loss on full set
                        ds = df.dropna(subset=args.features).reset_index(drop=True)
                        ds_model = model.eval()
                        import torch
                        xs = torch.tensor(ds[args.features].values.astype(float), dtype=torch.float32)
                        with torch.no_grad():
                            logits = ds_model(xs)
                            lossfn = torch.nn.BCEWithLogitsLoss()
                            ys = torch.tensor((ds['winner']=='Red').astype(float)).unsqueeze(1)
                            if label_invert:
                                ys = 1 - ys
                            loss_val = lossfn(logits, ys).item()
                        print(f'    final loss: {loss_val:.4f}')
                        if best is None or loss_val < best:
                            best = loss_val
                            best_cfg = (lr, batch, epochs)
            print(f'BEST configuration (invert={label_invert}): lr={best_cfg[0]}, batch={best_cfg[1]}, epochs={best_cfg[2]} -> loss={best:.4f}')
        else:
            if args.auto_lr:
                from model.neural_network import find_learning_rate
                print('running learning-rate finder...')
                lrs, losses, best_lr = find_learning_rate(df, args.features,
                                                          batch_size=args.batch,
                                                          invert=label_invert)
                print(f'suggested lr = {best_lr:.6g}')
                args.lr = best_lr
            model = UFCPredictor(input_dim=len(args.features))
            train_model(model, df, args.features,
                        epochs=args.epochs, lr=args.lr, batch_size=args.batch,
                        invert=label_invert,
                        weight_decay=args.weight_decay,
                        patience=args.patience)
            if prefix:
                import torch
                # make sure the parent directory exists
                import os
                outpath = prefix + ("_loss.pt" if label_invert else "_win.pt")
                os.makedirs(os.path.dirname(outpath), exist_ok=True)
                torch.save(model.state_dict(), outpath)
                print(f"saved model to {outpath}")

    # perform training runs
    if args.double:
        do_train(label_invert=False, prefix=args.save or 'model/checkpoints/model')
        do_train(label_invert=True, prefix=args.save or 'model/checkpoints/model')
    else:
        do_train(label_invert=args.invert, prefix=args.save or 'model/checkpoints/model')


if __name__ == '__main__':
    main()

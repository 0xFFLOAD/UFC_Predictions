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
import torch
import numpy as np
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
    parser.add_argument('--hidden3', type=int, default=0,
                        help='Size of optional third hidden layer (0 disables)')
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
    parser.add_argument('--hidden1-values', type=str,
                        help='Comma-separated list of hidden1 sizes for grid search')
    parser.add_argument('--hidden2-values', type=str,
                        help='Comma-separated list of hidden2 sizes for grid search')
    parser.add_argument('--hidden3-values', type=str,
                        help='Comma-separated list of hidden3 sizes for grid search')
    parser.add_argument('--seed-values', type=str,
                        help='Comma-separated list of random seeds to test during search')
    parser.add_argument('--auto-lr', action='store_true',
                        help='Run a learning-rate finder before training')
    parser.add_argument('--per-class', action='store_true',
                        help='Train separate models for each weight_class present in the data')
    parser.add_argument('--invert', action='store_true',
                        help='Flip labels (Red->0, Blue->1), i.e. predict loss instead of win')
    parser.add_argument('--double', action='store_true',
                        help='Train both win and loss models sequentially')
    parser.add_argument('--joint', action='store_true',
                        help='Train a single model on both win and inverted-loss examples (data duplicated)')
    parser.add_argument('--save', type=str,
                        help='Path prefix to save trained model(s); extension and suffix appended automatically')
    parser.add_argument('--ensemble', type=int, default=1,
                        help='Train this many models and ensemble their outputs')
    args = parser.parse_args()
    # debug: display key hyperparameters
    print(f"parsed args.hidden1={args.hidden1}, hidden2={args.hidden2}, hidden3={args.hidden3}, dropout={args.dropout}")

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
        # start with the first frame and merge each subsequent one, but
        # only bring in columns that are not already present to avoid
        # collisions (e.g. multiple copies of r_age/b_age).
        df = dfs[0]
        for other in dfs[1:]:
            on_cols = ['r_fighter', 'b_fighter', 'winner']
            if 'weight_class' in df.columns and 'weight_class' in other.columns:
                on_cols.append('weight_class')
            # select new feature columns only
            new_cols = [c for c in other.columns if c not in df.columns or c in on_cols]
            if not new_cols:
                continue
            df = df.merge(other[new_cols], on=on_cols, how='inner')
    # if the user requested joint training, duplicate rows with flipped labels
    if args.joint:
        if args.invert or args.double:
            parser.error('--joint cannot be used with --invert or --double')
        inv = df.copy()
        inv['winner'] = inv['winner'].apply(lambda w: 'Red' if w == 'Blue' else 'Blue')
        df = pd.concat([df, inv], ignore_index=True)
        print(f"joint mode: dataset size doubled to {len(df)} rows")

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
            h1_list = parse_list(args.hidden1_values, int) or [args.hidden1]
            h2_list = parse_list(args.hidden2_values, int) or [args.hidden2]
            h3_list = parse_list(args.hidden3_values, int) or [args.hidden3]
            seeds = parse_list(args.seed_values, int) or [None]
            if not (lrs and batches and epochs_list):
                parser.error('Search mode requires --lr-values, --batch-values, and --epoch-values')

            best = None
            best_cfg = None
            import torch
            for lr in lrs:
                for batch in batches:
                    for epochs in epochs_list:
                        for h1 in h1_list:
                            for h2 in h2_list:
                                for h3 in h3_list:
                                    for seed in seeds:
                                        if seed is not None:
                                            torch.manual_seed(seed)
                                        model = UFCPredictor(input_dim=len(args.features),
                                                             hidden1=h1,
                                                             hidden2=h2,
                                                             hidden3=h3,
                                                             dropout=args.dropout)
                                        print(f"-> testing lr={lr}, batch={batch}, epochs={epochs} hidden1={h1} hidden2={h2} hidden3={h3} seed={seed} invert={label_invert}")
                                        train_model(model, df, args.features,
                                                    epochs=epochs, lr=lr, batch_size=batch,
                                                    invert=label_invert,
                                                    weight_decay=args.weight_decay,
                                                    patience=args.patience)
                                        # save this particular configuration so every trial is
                                        # persisted; include hyperparameters and timestamp to
                                        # avoid collisions and allow later inspection
                                        if prefix:
                                            import time, os
                                            ts = int(time.time() * 1000)
                                            cfgname = f"lr{lr}_b{batch}_e{epochs}_h1{h1}_h2{h2}_h3{h3}_s{seed}_{ts}"
                                            outpath = prefix + "_" + cfgname + ("_loss.pt" if label_invert else "_win.pt")
                                            os.makedirs(os.path.dirname(outpath), exist_ok=True)
                                            torch.save(model.state_dict(), outpath)
                                            print(f"    saved trial to {outpath}")
                                        # after training compute loss on clean dataset using same
                                        # normalization that was applied during training.  the
                                        # previous implementation fed raw feature values to the
                                        # model, which produced wildly different numbers.  here we
                                        # instantiate a FeatureDataset and reuse its tensors.
                                        import torch
                                        from model.neural_network import FeatureDataset
                                        ds_clean = df.dropna(subset=args.features).reset_index(drop=True)
                                        if len(ds_clean) > 0:
                                            dataset_obj = FeatureDataset(ds_clean, args.features,
                                                                        invert=label_invert)
                                            xs = dataset_obj.features.to(next(model.parameters()).device)
                                            ys = dataset_obj.labels.to(xs.device)
                                            model.eval()
                                            with torch.no_grad():
                                                logits = model(xs)
                                                lossfn = torch.nn.BCEWithLogitsLoss()
                                                loss_val = lossfn(logits, ys).item()
                                        else:
                                            loss_val = float('inf')
                                        print(f'    final loss: {loss_val:.4f}')
                                        if best is None or loss_val < best:
                                            best = loss_val
                                            best_cfg = (lr, batch, epochs, h1, h2, h3, seed)
            print(f"BEST configuration (invert={label_invert}): lr={best_cfg[0]}, batch={best_cfg[1]}, epochs={best_cfg[2]}, hidden1={best_cfg[3]}, hidden2={best_cfg[4]}, hidden3={best_cfg[5]}, seed={best_cfg[6]} -> loss={best:.4f}")
        else:
            if args.auto_lr:
                from model.neural_network import find_learning_rate
                print('running learning-rate finder...')
                lrs, losses, best_lr = find_learning_rate(df, args.features,
                                                          batch_size=args.batch,
                                                          invert=label_invert)
                print(f'suggested lr = {best_lr:.6g}')
                args.lr = best_lr
            # instantiate with specified architecture (args may have been updated by auto-lr)
            model = UFCPredictor(input_dim=len(args.features),
                                 hidden1=args.hidden1,
                                 hidden2=args.hidden2,
                                 dropout=args.dropout)
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

    # perform training runs (possibly ensemble)
    def run_seq(label_invert, base_prefix):
        import torch, os
        # we no longer delete previous files; every run will produce
        # a distinct checkpoint.  the `do_train` helper adds a timestamp
        # or hyperparameter suffix to guarantee uniqueness.
        paths = []
        for i in range(args.ensemble):
            seed = None if args.ensemble == 1 else i
            prefix = base_prefix + (f"_{i}" if args.ensemble>1 else "")
            if seed is not None:
                torch.manual_seed(seed)
            do_train(label_invert=label_invert, prefix=prefix)
            # the helper returns the actual path it saved (it appends .pt)
            paths.append(prefix + ("_loss.pt" if label_invert else "_win.pt"))
        return paths

    if args.double:
        win_models = run_seq(False, args.save or 'model/checkpoints/model')
        loss_models = run_seq(True, args.save or 'model/checkpoints/model')
    else:
        models = run_seq(args.invert, args.save or 'model/checkpoints/model')
        win_models = models if not args.invert else []
        loss_models = models if args.invert else []

    # if ensemble more than one, evaluate accuracy on training data
    if args.ensemble > 1:
        import torch
        df_full = df.dropna(subset=args.features).reset_index(drop=True)
        import numpy as np
        def eval_models(paths, invert_flag):
            xs = torch.tensor(df_full[args.features].values.astype(float), dtype=torch.float32)
            ys = torch.tensor((df_full.loc[:, 'winner']=='Red').astype(float)).unsqueeze(1)
            if invert_flag:
                ys = 1 - ys
            logits_sum = None
            valid = 0
            for p in paths:
                m = UFCPredictor(input_dim=len(args.features),
                                  hidden1=args.hidden1,
                                  hidden2=args.hidden2,
                                  hidden3=args.hidden3,
                                  dropout=args.dropout)
                try:
                    m.load_state_dict(torch.load(p))
                except Exception as e:
                    print(f'warning: skipping ensemble member {p} ({e})')
                    continue
                m.eval()
                with torch.no_grad():
                    out = torch.sigmoid(m(xs))
                logits_sum = out if logits_sum is None else logits_sum + out
                valid += 1
            if valid == 0:
                raise RuntimeError('no valid models in ensemble')
            # average only over successful members
            logits_sum = logits_sum / valid
            pred_labels = (logits_sum > 0.5).float()
            acc = (pred_labels == ys).float().mean().item()
            return acc
        if win_models:
            acc = eval_models(win_models, invert_flag=False)
            print(f'Ensemble accuracy (win models) on training set: {acc*100:.2f}%')
        if loss_models:
            acc = eval_models(loss_models, invert_flag=True)
            print(f'Ensemble accuracy (loss models) on training set: {acc*100:.2f}%')


if __name__ == '__main__':
    main()

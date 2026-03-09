"""Train a single network on each weight-class TSV and persist weights.

This helper lives alongside the age extractor and automates the
per-class workflow.  It will:

1. if no explicit data sources are given, scan the current directory for
   files matching ``age_*.tsv`` (excluding ``age.tsv`` itself).  you can
   alternatively pass ``--data`` with one or more TSV paths.
2. for each weight class, load the corresponding rows from every source
   and merge them on the fighter identifiers.  this allows you to
   combine features (e.g. age plus age-delta) without manual preprocessing.
3. train a fresh `UFCPredictor` using the standard training script
   logic (with optional auto-LR), again using the same column list for
   every class.
4. save the model's state dict under `../../model/checkpoints/<class>.pt`
   so the learned parameters can be reused later without retraining

The script is idempotent: if a checkpoint for a class already exists it
is skipped unless ``--force`` is passed.

Usage::

    cd extract/age
    python train_by_class.py --epochs 50 --batch 32 --auto-lr

The same set of features (by default ``['r_age','b_age']``) is used for
all classes.  You can override via the ``--features`` argument."""

import argparse
import glob
import os
import pandas as pd

from model.neural_network import UFCPredictor, train_model

CHECKPOINT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              '..', 'model', 'checkpoints'))
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Train per-class age models")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden1', type=int, default=64,
                        help='Size of first hidden layer')
    parser.add_argument('--hidden2', type=int, default=32,
                        help='Size of second hidden layer')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout probability')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='L2 regularization strength')
    parser.add_argument('--patience', type=int,
                        help='Early stopping patience')
    parser.add_argument('--auto-lr', action='store_true',
                        help='Run LR finder before each class')

    # hyperparameter search options (mirrors --search in model/train.py)
    parser.add_argument('--search', action='store_true',
                        help='Perform grid search over hyperparameters per class')
    parser.add_argument('--lr-values',
                        help='Comma-separated list of lr values for grid search')
    parser.add_argument('--batch-values',
                        help='Comma-separated list of batch sizes for grid search')
    parser.add_argument('--epoch-values',
                        help='Comma-separated list of epoch counts for grid search')
    parser.add_argument('--hidden1-values',
                        help='Comma-separated list of hidden1 sizes for grid search')
    parser.add_argument('--hidden2-values',
                        help='Comma-separated list of hidden2 sizes for grid search')
    parser.add_argument('--hidden3-values',
                        help='Comma-separated list of hidden3 sizes for grid search')
    parser.add_argument('--seed-values',
                        help='Comma-separated list of random seeds to test during search')

    parser.add_argument('--features', nargs='+', default=['r_age', 'b_age'],
                        help='Feature columns to use')
    parser.add_argument('--data', nargs='+',
                        help='TSV files to merge for each class; if omitted will use age_*.tsv in the current directory')
    parser.add_argument('--invert', action='store_true',
                        help='Train models predicting loss instead of win for each class')
    parser.add_argument('--double', action='store_true',
                        help='Train both win and loss models for each class')
    parser.add_argument('--force', action='store_true',
                        help='Retrain even if checkpoint exists')
    args = parser.parse_args()

    # determine how we will obtain data for a given class
    if args.data:
        # explicit feature tables -> merge them per class
        source_files = args.data

        # helper used by get_class_df below
        def load_for_class(path, cls):
            df = pd.read_csv(path, sep='\t')
            if 'weight_class' in df.columns:
                df = df[df['weight_class'] == cls]
            return df

        def get_class_df(cls):
            dfs = [load_for_class(path, cls) for path in source_files]
            df = dfs[0]
            for other in dfs[1:]:
                on_cols = ['r_fighter', 'b_fighter', 'winner']
                if 'weight_class' in df.columns and 'weight_class' in other.columns:
                    on_cols.append('weight_class')
                df = df.merge(other, on=on_cols, how='inner')
            return df

        # get classes from first source without filtering
        first = pd.read_csv(source_files[0], sep='\t')
        if first.empty or 'weight_class' not in first.columns:
            print('no data sources found, exiting')
            return
        classes = first['weight_class'].dropna().unique()
    else:
        # no explicit sources -> use per-class TSVs directly
        pattern = os.path.join(os.path.dirname(__file__), 'age_*.tsv')
        source_files = [f for f in glob.glob(pattern) if os.path.basename(f) != 'age.tsv']

        def get_class_df(cls):
            path = os.path.join(os.path.dirname(__file__), f'age_{cls}.tsv')
            return pd.read_csv(path, sep='\t')

        # classes are derived from the filenames themselves
        classes = [os.path.splitext(os.path.basename(f))[0].replace('age_', '')
                   for f in source_files]

    for cls in classes:
        # determine file basenames for this class depending on invert/double
        def checkpoint_name(invert_flag):
            base = f'{cls}'
            if args.double:
                suffix = '_loss' if invert_flag else '_win'
            elif args.invert:
                suffix = '_loss'
            else:
                suffix = ''
            return os.path.join(CHECKPOINT_DIR, base + suffix + '.pt')

        # function to train one model
        def run_single(invert_flag):
            chk = checkpoint_name(invert_flag)
            if os.path.exists(chk) and not args.force:
                print(f"Skipping {cls} ({'loss' if invert_flag else 'win'}), checkpoint already exists")
                return
            print(f"Preparing data for class {cls} (invert={invert_flag})")
            df = get_class_df(cls)

            # ensure the requested features actually exist in this class frame
            avail = [f for f in args.features if f in df.columns]
            if not avail:
                print(f"  class {cls} has none of features {args.features}, skipping")
                return
            df_clean = df.dropna(subset=avail)
            print(f"Training class {cls} ({len(df_clean)} rows after merge, using {avail})")
            if len(df_clean) == 0:
                print(f"  no data after dropping NaNs, skipping {cls}")
                return
            model = UFCPredictor(input_dim=len(args.features),
                                 hidden1=args.hidden1,
                                 hidden2=args.hidden2,
                                 dropout=args.dropout)

            # determine learning rate, either via auto-lr or fixed
            if not args.search:
                if args.auto_lr and len(df_clean) >= args.batch:
                    from model.neural_network import find_learning_rate
                    print('  finding learning rate...')
                    lrs, losses, best_lr = find_learning_rate(df_clean, args.features,
                                                              batch_size=args.batch,
                                                              invert=invert_flag)
                    if best_lr is None:
                        best_lr = args.lr
                    print(f'  suggested lr {best_lr:.6g}')
                    lr_val = best_lr
                else:
                    lr_val = args.lr

                train_model(model, df_clean, args.features,
                            epochs=args.epochs, lr=lr_val, batch_size=args.batch,
                            invert=invert_flag,
                            weight_decay=args.weight_decay,
                            patience=args.patience)

                torch = __import__('torch')
                torch.save(model.state_dict(), chk)
                print(f"Saved checkpoint to {chk}\n")
            else:
                # perform exhaustive grid search for this class
                def parse_list(s, cast):
                    return [cast(x) for x in s.split(',')] if s else []
                lrs = parse_list(args.lr_values, float)
                batches = parse_list(args.batch_values, int)
                epochs_list = parse_list(args.epoch_values, int)
                h1_list = parse_list(args.hidden1_values, int) or [args.hidden1]
                h2_list = parse_list(args.hidden2_values, int) or [args.hidden2]
                h3_list = parse_list(args.hidden3_values, int) or [0]
                seeds = parse_list(args.seed_values, int) or [None]
                if not (lrs and batches and epochs_list):
                    parser.error('Search mode requires --lr-values, --batch-values, and --epoch-values')

                best = None
                best_cfg = None
                import torch
                best_model = None
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
                                            print(f"-> class {cls}: testing lr={lr}, batch={batch}, epochs={epochs} hidden1={h1} hidden2={h2} hidden3={h3} seed={seed} invert={invert_flag}")
                                            train_model(model, df_clean, args.features,
                                                        epochs=epochs, lr=lr, batch_size=batch,
                                                        invert=invert_flag,
                                                        weight_decay=args.weight_decay,
                                                        patience=args.patience)
                                            # compute loss on full cleaned set using the same
                                            # normalization as training; previous code fed raw
                                            # values, which was why `epoch` loss sometimes didn't
                                            # match the reported `final loss`.
                                            from model.neural_network import FeatureDataset
                                            ds = df_clean.reset_index(drop=True)
                                            if len(ds) > 0:
                                                dataset_obj = FeatureDataset(ds, args.features,
                                                                            invert=invert_flag)
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
                                                best_model = model
                print(f"BEST configuration for class {cls} (invert={invert_flag}): lr={best_cfg[0]}, batch={best_cfg[1]}, epochs={best_cfg[2]}, hidden1={best_cfg[3]}, hidden2={best_cfg[4]}, hidden3={best_cfg[5]}, seed={best_cfg[6]} -> loss={best:.4f}")
                if best_model is not None:
                    torch.save(best_model.state_dict(), chk)
                    print(f"Saved best checkpoint to {chk}\n")


        # decide which runs to perform
        if args.double:
            run_single(False)
            run_single(True)
        elif args.invert:
            run_single(True)
        else:
            run_single(False)


if __name__ == '__main__':
    main()

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
    parser.add_argument('--auto-lr', action='store_true',
                        help='Run LR finder before each class')
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

    # determine source files for each class
    if args.data:
        source_files = args.data
    else:
        pattern = os.path.join(os.path.dirname(__file__), 'age_*.tsv')
        source_files = [f for f in glob.glob(pattern) if os.path.basename(f) != 'age.tsv']

    # helper to read and optionally filter to class
    def load_for_class(path, cls):
        df = pd.read_csv(path, sep='\t')
        if 'weight_class' in df.columns:
            df = df[df['weight_class'] == cls]
        return df

    # gather all classes by inspecting the first source (no filtering)
    if not source_files:
        print('no data sources found, exiting')
        return
    first = pd.read_csv(source_files[0], sep='\t')
    if first.empty or 'weight_class' not in first.columns:
        print('no data sources found, exiting')
        return
    classes = first['weight_class'].dropna().unique()

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
            # merge all sources for this class horizontally
            dfs = [load_for_class(path, cls) for path in source_files]
            df = dfs[0]
            for other in dfs[1:]:
                on_cols = ['r_fighter', 'b_fighter', 'winner']
                if 'weight_class' in df.columns and 'weight_class' in other.columns:
                    on_cols.append('weight_class')
                df = df.merge(other, on=on_cols, how='inner')

            df_clean = df.dropna(subset=args.features)
            print(f"Training class {cls} ({len(df_clean)} rows after merge)")
            if len(df_clean) == 0:
                print(f"  no data after dropping NaNs, skipping {cls}")
                return
            model = UFCPredictor(input_dim=len(args.features))

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
                        invert=invert_flag)

            torch = __import__('torch')
            torch.save(model.state_dict(), chk)
            print(f"Saved checkpoint to {chk}\n")

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

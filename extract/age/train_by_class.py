"""Train a single network on each weight-class TSV and persist weights.

This helper lives alongside the age extractor and automates the
per-class workflow.  It will:

1. scan the current directory for files matching ``age_*.tsv`` (excluding
   ``age.tsv`` itself)
2. for each file train a fresh `UFCPredictor` using the standard
   training script logic (with optional auto-LR)
3. save the model's state dict under `../../model/checkpoints/<class>.pt`
   so the learned parameters can be reused later without retraining

The script is idempotent: if a checkpoint for a class already exists it
is skipped unless ``--force`` is passed.

Usage::

    cd extract/age
    python train_by_class.py --epochs 50 --batch 32 --auto-lr

The same set of features (by default ``['r_age','b_age']``) is used for
all classes.  You can override via the ``--features`` argument.
"""

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
    parser.add_argument('--force', action='store_true',
                        help='Retrain even if checkpoint exists')
    args = parser.parse_args()

    pattern = os.path.join(os.path.dirname(__file__), 'age_*.tsv')
    files = glob.glob(pattern)
    files = [f for f in files if os.path.basename(f) != 'age.tsv']

    for path in files:
        cls = os.path.splitext(os.path.basename(path))[0].replace('age_', '')
        chk = os.path.join(CHECKPOINT_DIR, f'{cls}.pt')
        if os.path.exists(chk) and not args.force:
            print(f"Skipping {cls}, checkpoint already exists")
            continue

        df = pd.read_csv(path, sep='\t')
        # drop rows with missing features
        df_clean = df.dropna(subset=args.features)
        # even a single row will be used; log size
        print(f"Training class {cls} from {path} ({len(df_clean)} rows)")
        if len(df_clean) == 0:
            print(f"  no data after dropping NaNs, skipping {cls}")
            continue
        model = UFCPredictor(input_dim=len(args.features))

        if args.auto_lr and len(df_clean) >= args.batch:
            from model.neural_network import find_learning_rate
            print('  finding learning rate...')
            # safe call: if finder returns no values, use default
            lrs, losses, best_lr = find_learning_rate(df_clean, args.features,
                                                      batch_size=args.batch)
            if best_lr is None:
                best_lr = args.lr
            print(f'  suggested lr {best_lr:.6g}')
            lr = best_lr
        else:
            # either auto-lr disabled or too few samples for a sweep
            lr = args.lr

        train_model(model, df_clean, args.features,
                    epochs=args.epochs, lr=lr, batch_size=args.batch)

        torch = __import__('torch')
        torch.save(model.state_dict(), chk)
        print(f"Saved checkpoint to {chk}\n")


if __name__ == '__main__':
    main()

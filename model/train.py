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
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()

    dfs = [pd.read_csv(f, sep='\t') for f in args.data]
    df = pd.concat(dfs, axis=0, ignore_index=True)

    model = UFCPredictor(input_dim=len(args.features))
    train_model(model, df, args.features,
                epochs=args.epochs, lr=args.lr, batch_size=args.batch)


if __name__ == '__main__':
    main()

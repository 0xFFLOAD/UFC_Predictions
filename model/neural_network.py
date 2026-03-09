"""A simple feed‑forward classifier for UFC fight prediction.

This module provides a PyTorch implementation of the architecture
mentioned in the README and a training helper.  The network is
initialized with an arbitrary number of input features so it can
later be trained on the 25 features extracted from the TSVs.

Example usage::

    from model.neural_network import UFCPredictor, train_model
    # df = pd.read_csv('extract/age/age.tsv')  # etc.
    features = ['height_diff', 'reach_diff', ...]
    model = UFCPredictor(input_dim=len(features))
    train_model(model, df, features)

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class FeatureDataset(Dataset):
    def __init__(self, dataframe, feature_columns, label_column='winner', invert: bool = False):
        # drop any rows where the requested features are missing
        df = dataframe.dropna(subset=feature_columns)
        arr = df[feature_columns].values.astype(float)
        # z-score normalize each column to zero mean/ unit variance
        # avoids huge logits when features have large range (like age)
        mean = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        normed = (arr - mean) / std
        # clip extreme values to avoid single outliers (e.g. weight delta)
        # dominating training; this keeps values within roughly 10 stddevs.
        import numpy as np
        normed = np.clip(normed, -10.0, 10.0)
        self.features = torch.tensor(normed, dtype=torch.float32)
        # convert winner to binary: 'Red' -> 1, 'Blue' -> 0 (or adjust)
        labels = (df[label_column] == 'Red').astype(float).values
        if invert:
            # predict loss: swap positive/negative
            labels = 1.0 - labels
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class UFCPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden1: int = 64, hidden2: int = 32,
                 hidden3: int = 0,
                 dropout: float = 0.0):
        super().__init__()
        # Build a sequential network; hidden3 is optional (0 disables it).
        layers = [nn.Linear(input_dim, hidden1), nn.Tanh()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend([nn.Linear(hidden1, hidden2), nn.Tanh()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        if hidden3 and hidden3 > 0:
            layers.extend([nn.Linear(hidden2, hidden3), nn.Tanh()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden3, 1))
        else:
            layers.append(nn.Linear(hidden2, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def find_learning_rate(df, feature_columns, label_column='winner',
                       init_lr=1e-6, final_lr=10, num_iters=100,
                       batch_size=32, device=None, invert: bool = False,
                       hidden1: int = 64, hidden2: int = 32, hidden3: int = 0,
                       dropout: float = 0.0):
    """Basic LR finder that returns a recommended learning rate.

    It trains the network for ``num_iters`` mini-batches, exponentially
    increasing the learning rate from ``init_lr`` to ``final_lr`` and
    records the loss.  The return value is ``(lrs, losses, best_lr)``
    where ``best_lr`` is the lr corresponding to the minimum loss.
    """
    import numpy as np

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UFCPredictor(input_dim=len(feature_columns),
                          hidden1=hidden1, hidden2=hidden2,
                          hidden3=hidden3, dropout=dropout).to(device)
    dataset = FeatureDataset(df, feature_columns, label_column,
                             invert=invert)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    criterion = nn.BCEWithLogitsLoss()

    lrs = []
    losses = []
    mult = (final_lr / init_lr) ** (1.0 / num_iters)
    it = 0
    for x_batch, y_batch in loader:
        if it >= num_iters:
            break
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(x_batch)
        if torch.isnan(preds).any() or torch.isinf(preds).any():
            break
        loss = criterion(preds, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())
        optimizer.param_groups[0]['lr'] *= mult
        it += 1
    if losses:
        best_idx = int(np.argmin(losses))
        best_lr = lrs[best_idx]
    else:
        best_lr = init_lr
    return lrs, losses, best_lr


def train_model(model: nn.Module,
                df,
                feature_columns,
                label_column='winner',
                epochs: int = 20,
                lr: float = 1e-3,
                batch_size: int = 32,
                device: str = None,
                invert: bool = False,
                weight_decay: float = 0.0,
                patience: int = None):
    """Train the model on the supplied DataFrame.

    Parameters
    ----------
    model : nn.Module
        The neural network instance to train.
    df : pandas.DataFrame
        Data containing features and a label column.
    feature_columns : list[str]
        Names of the columns to use as inputs.
    label_column : str, optional
        Column containing the winner string, by default 'winner'.
    epochs : int, optional
        Number of training epochs, by default 20.
    lr : float, optional
        Learning rate, by default 1e-3.
    batch_size : int, optional
        Mini-batch size, by default 32.
    device : str | None, optional
        Torch device; if None will select cuda if available.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    dataset = FeatureDataset(df, feature_columns, label_column)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # use logits loss to avoid manual sigmoid and range issues
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    model.train()
    best_loss = float('inf')
    patience_cnt = 0
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            # check for NaN/inf in preds
            if torch.isnan(preds).any() or torch.isinf(preds).any():
                raise RuntimeError("Model produced NaN/Inf outputs; try smaller lr or add clipping")
            loss = criterion(preds, y_batch)
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f}")
        # early stopping
        if patience is not None:
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    print(f"Early stopping after {epoch} epochs (no improvement for {patience} epochs)")
                    break


if __name__ == '__main__':
    print('Neural network module. Import UFCPredictor and train_model.')

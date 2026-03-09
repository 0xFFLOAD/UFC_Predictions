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
    def __init__(self, dataframe, feature_columns, label_column='winner'):
        self.features = torch.tensor(
            dataframe[feature_columns].values, dtype=torch.float32
        )
        # convert winner to binary: 'Red' -> 1, 'Blue' -> 0 (or adjust)
        self.labels = torch.tensor(
            (dataframe[label_column] == 'Red').astype(float).values,
            dtype=torch.float32,
        ).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class UFCPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden1: int = 64, hidden2: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.Tanh(),
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train_model(model: nn.Module,
                df,
                feature_columns,
                label_column='winner',
                epochs: int = 20,
                lr: float = 1e-3,
                batch_size: int = 32,
                device: str = None):
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

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f}")


if __name__ == '__main__':
    print('Neural network module. Import UFCPredictor and train_model.')

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch import nn

np.random.seed(0)


def one_hot_e(dna_seq: str) -> np.ndarray:
  mydict = {'A': np.asarray([1.0, 0.0, 0.0, 0.0]), 'C': np.asarray([0.0, 1.0, 0.0, 0.0]),
            'G': np.asarray([0.0, 0.0, 1.0, 0.0]), 'T': np.asarray([0.0, 0.0, 0.0, 1.0]),
            'N': np.asarray([0.0, 0.0, 0.0, 0.0]), 'H': np.asarray([0.0, 0.0, 0.0, 0.0]),
            'a': np.asarray([1.0, 0.0, 0.0, 0.0]), 'c': np.asarray([0.0, 1.0, 0.0, 0.0]),
            'g': np.asarray([0.0, 0.0, 1.0, 0.0]), 't': np.asarray([0.0, 0.0, 0.0, 1.0]),
            'n': np.asarray([0.0, 0.0, 0.0, 0.0]), '-': np.asarray([0.0, 0.0, 0.0, 0.0])}

  size_of_a_seq: int = len(dna_seq)

  # forward = np.zeros(shape=(size_of_a_seq, 4))

  forward_list: list = [mydict[dna_seq[i]] for i in range(0, size_of_a_seq)]
  encoded = np.asarray(forward_list)
  return encoded


#
# Load and prepare data    "CTCATGTCA"
#
df = pd.read_csv('old2/data64random.csv')

# To numpy arrays, and encode X
X = np.stack([one_hot_e(row) for row in df.Sequence], axis=0)
y = df['class'].values

# Shuffle and split
train_size = int(0.6 * len(X))
val_size = int(0.3 * len(X))

shuffle_ixs = np.random.permutation(len(X))
X, y = [arr[shuffle_ixs] for arr in [X, y]]

X_train, y_train = [arr[:train_size] for arr in [X, y]]
X_val, y_val = [arr[train_size:train_size + val_size] for arr in [X, y]]

# As tensors. Useful for passing directly to model as a single large batch.
X_train_t, y_train_t = [torch.tensor(arr).float() for arr in [X_train, y_train]]
X_val_t, y_val_t = [torch.tensor(arr).float() for arr in [X_val, y_val]]


#
# Define the model
#

# Lambda layer useful for simple manipulations
class LambdaLayer(nn.Module):
  def __init__(self, func):
    super().__init__()
    self.func = func

  def forward(self, x):
    return self.func(x)


# batch, 64, chan

# The model
seq_len = X[0].shape[0]  # 64 characters long
n_features = X[0].shape[1]  # 4-dim onehot encoding

torch.manual_seed(0)
model = nn.Sequential(
  # > (batch, seq_len, channels)

  LambdaLayer(lambda x: x.swapdims(1, 2)),
  # > (batch, channels, seq_len)

  # Initial wide receptive field (and it matches length of the pattern)
  nn.Conv1d(in_channels=n_features, out_channels=4, kernel_size=9, padding='same'),
  nn.ReLU(),
  nn.BatchNorm1d(num_features=4),

  # Conv block 1 doubles features
  nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3),
  nn.ReLU(),
  nn.BatchNorm1d(num_features=8),

  # Conv block 2, then maxpool
  nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3),
  nn.ReLU(),
  nn.BatchNorm1d(num_features=8),

  # Output layer: flatten, linear
  nn.MaxPool1d(kernel_size=2, stride=2),  # batch, feat, seq
  nn.Flatten(start_dim=1),  # batch, feat*seq
  nn.Linear(8 * 30, 1),
)
print(
  'Model size is',
  sum([p.numel() for p in model.parameters() if p.requires_grad]),
  'trainable parameters'
)

# Train loader for batchifying train data
train_loader = DataLoader(list(zip(X_train_t, y_train_t)), shuffle=True, batch_size=8)

optimiser = torch.optim.NAdam(model.parameters())
loss_fn = nn.BCEWithLogitsLoss()

from collections import defaultdict

metrics_dict = defaultdict(list)

for epoch in range(n_epochs := 15):
  model.train()
  cum_loss = 0

  for X_minibatch, y_minibatch in train_loader:
    logits = model(X_minibatch).ravel()
    loss = loss_fn(logits, y_minibatch)

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    cum_loss += loss.item() * len(X_minibatch)
  # /end of epoch

  time_to_print = (epoch == 0) or ((epoch + 1) % 1) == 0
  if not time_to_print:
    continue

  model.eval()

  with torch.no_grad():
    val_logits = model(X_val_t).ravel()
    trn_logits = model(X_train_t).ravel()

  val_loss = loss_fn(val_logits, y_val_t).item()
  trn_loss = cum_loss / len(X_train_t)

  val_acc = ((nn.Sigmoid()(val_logits) > 0.5).int() == y_val_t.int()).float().mean().item()
  trn_acc = ((nn.Sigmoid()(trn_logits) > 0.5).int() == y_train_t.int()).float().mean().item()
  print(
    f'[epoch {epoch + 1:>3d}]',
    f'trn loss: {trn_loss:>5.3f} [acc: {trn_acc:>8.3%}] |',
    f'val loss: {val_loss:>5.3f} [acc: {val_acc:>8.3%}]'
  )

  # Record metrics
  metrics_dict['epoch'].append(epoch + 1)
  metrics_dict['trn_loss'].append(trn_loss)
  metrics_dict['val_loss'].append(val_loss)
  metrics_dict['trn_acc'].append(trn_acc)
  metrics_dict['val_acc'].append(val_acc)

# View training curves
metrics_df = pd.DataFrame(metrics_dict).set_index('epoch')
ax = metrics_df.plot(
  use_index=True, y=['trn_loss', 'val_loss'], ylabel='loss',
  figsize=(8, 4), legend=False, linewidth=3, marker='s', markersize=7
)

metrics_df.mul(100).plot(
  use_index=True, y=['trn_acc', 'val_acc'], ylabel='acc',
  linestyle='--', marker='o', ax=ax.twinx(), legend=False
)
ax.figure.legend(ncol=2)
ax.set_title('training curves')
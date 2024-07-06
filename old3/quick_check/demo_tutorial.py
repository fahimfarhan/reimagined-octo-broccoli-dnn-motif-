import random

import numpy as np
import pandas as pd
import torch
from itertools import product

from torch import nn


def kmers(k):
  '''Generate a list of all k-mers for a given k'''

  return [''.join(x) for x in product(['A', 'C', 'G', 'T'], repeat=k)]


score_dict = {
  'A': 20,
  'C': 17,
  'G': 14,
  'T': 11
}


def score_seqs_motif(seqs):
  '''
  Calculate the scores for a list of sequences based on
  the above score_dict
  '''
  data = []
  for seq in seqs:
    # get the average score by nucleotide
    score = np.mean([score_dict[base] for base in seq])

    # give a + or - bump if this k-mer has a specific motif
    if 'TAT' in seq:
      score += 10
    if 'GCG' in seq:
      score -= 10
    data.append([seq, score])

  df = pd.DataFrame(data, columns=['seq', 'score'])
  return df


seqs8 = kmers(8)
mer8 = score_seqs_motif(seqs8)


def one_hot_encode(seq):
  """
  Given a DNA sequence, return its one-hot encoding
  """
  # Make sure seq has only allowed bases
  allowed = set("ACTGN")
  if not set(seq).issubset(allowed):
    invalid = set(seq) - allowed
    raise ValueError(f"Sequence contains chars not in allowed DNA alphabet (ACGTN): {invalid}")

  # Dictionary returning one-hot encoding for each nucleotide
  nuc_d = {'A': [1.0, 0.0, 0.0, 0.0],
           'C': [0.0, 1.0, 0.0, 0.0],
           'G': [0.0, 0.0, 1.0, 0.0],
           'T': [0.0, 0.0, 0.0, 1.0],
           'N': [0.0, 0.0, 0.0, 0.0]}

  # Create array from nucleotide sequence
  vec = np.array([nuc_d[x] for x in seq])

  return vec


# look at DNA seq of 8 As
a8 = one_hot_encode("AAAAAAAA")
print("AAAAAAAA:\n", a8)

# prints:
# AAAAAAAA:
# [[1. 0. 0. 0.]
# [1. 0. 0. 0.]
# [1. 0. 0. 0.]
# [1. 0. 0. 0.]
# [1. 0. 0. 0.]
# [1. 0. 0. 0.]
# [1. 0. 0. 0.]
# [1. 0. 0. 0.]]

# look at DNA seq of random nucleotides
s = one_hot_encode("AGGTACCT")
print("AGGTACCT:\n", s)
print("shape:", s.shape)


# prints:
# AGGTACCT:
# [[1. 0. 0. 0.]
# [0. 0. 1. 0.]
# [0. 0. 1. 0.]
# [0. 0. 0. 1.]
# [1. 0. 0. 0.]
# [0. 1. 0. 0.]
# [0. 1. 0. 0.]
# [0. 0. 0. 1.]]
# shape: (8, 4)


def quick_split(df, split_frac=0.8):
  '''
  Given a df of samples, randomly split indices between
  train and test at the desired fraction
  '''
  cols = df.columns  # original columns, use to clean up reindexed cols
  df = df.reset_index()

  # shuffle indices
  idxs = list(range(df.shape[0]))
  random.shuffle(idxs)

  # split shuffled index list by split_frac
  split = int(len(idxs) * split_frac)
  train_idxs = idxs[:split]
  test_idxs = idxs[split:]

  # split dfs and return
  train_df = df[df.index.isin(train_idxs)]
  test_df = df[df.index.isin(test_idxs)]

  return train_df[cols], test_df[cols]


full_train_df, test_df = quick_split(mer8)
train_df, val_df = quick_split(full_train_df)

print("Train:", train_df.shape)
print("Val:", val_df.shape)
print("Test:", test_df.shape)

# prints:
# Train: (41942, 2)
# Val: (10486, 2)
# Test: (13108, 2)


from torch.utils.data import Dataset, DataLoader


## Here is a custom defined Dataset object specialized for one-hot encoded DNA:

class SeqDatasetOHE(Dataset):
  '''
  Dataset for one-hot-encoded sequences
  '''

  def __init__(self,
               df,
               seq_col='seq',
               target_col='score'
               ):
    # +--------------------+
    # | Get the X examples |
    # +--------------------+
    # extract the DNA from the appropriate column in the df
    self.seqs = list(df[seq_col].values)
    self.seq_len = len(self.seqs[0])

    # one-hot encode sequences, then stack in a torch tensor
    self.ohe_seqs = torch.stack([torch.tensor(one_hot_encode(x)) for x in self.seqs])

    # +------------------+
    # | Get the Y labels |
    # +------------------+
    self.labels = torch.tensor(list(df[target_col].values)).unsqueeze(1)

  def __len__(self): return len(self.seqs)

  def __getitem__(self, idx):
    # Given an index, return a tuple of an X with it's associated Y
    # This is called inside DataLoader
    seq = self.ohe_seqs[idx]
    label = self.labels[idx]

    return seq, label


## Here is how I constructed DataLoaders from Datasets.

def build_dataloaders(train_df,
                      test_df,
                      seq_col='seq',
                      target_col='score',
                      batch_size=128,
                      shuffle=True
                      ):
  '''
  Given a train and test df with some batch construction
  details, put them into custom SeqDatasetOHE() objects.
  Give the Datasets to the DataLoaders and return.
  '''

  # create Datasets
  train_ds = SeqDatasetOHE(train_df, seq_col=seq_col, target_col=target_col)
  test_ds = SeqDatasetOHE(test_df, seq_col=seq_col, target_col=target_col)

  # Put DataSets into DataLoaders
  train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
  test_dl = DataLoader(test_ds, batch_size=batch_size)

  return train_dl, test_dl


train_dl, val_dl = build_dataloaders(train_df, val_df)

# basic CNN model
class DNA_CNN(nn.Module):
  def __init__(self,
               seq_len,
               num_filters=32,
               kernel_size=3):
    super().__init__()
    self.seq_len = seq_len

    self.conv_net = nn.Sequential(
      # 4 is for the 4 nucleotides
      nn.Conv1d(4, num_filters, kernel_size=kernel_size),
      nn.ReLU(inplace=True),
      nn.Flatten(),
      nn.Linear(num_filters * (seq_len - kernel_size + 1), 1)
    )

  def forward(self, xb):
    # permute to put channel in correct order
    # (batch_size x 4channel x seq_len)
    xb = xb.permute(0, 2, 1)
    # print(xb.shape)
    out = self.conv_net(xb)
    print(f"{xb.shape = }, {out.shape = }")
    return out


def start():
  print(f"{mer8 =}")
  dna_cnn = DNA_CNN(seq_len=8).double()

  for data in train_dl:
    seq, score = data
    output = dna_cnn(seq)
    print(f" { seq.shape = }, { score.shape = }, { output.shape = }")


if __name__ == '__main__':
  start()
  pass
# INFO:root:ohe.shape = torch.Size([16, 4, 4000]),  label.shape = torch.Size([16]),  output.shape = torch.Size([1, 1])
#   seq.shape = torch.Size([128, 8, 4]),  score.shape = torch.Size([128, 1])
#   xb.shape = torch.Size([86, 4, 8]), out.shape = torch.Size([86, 1])

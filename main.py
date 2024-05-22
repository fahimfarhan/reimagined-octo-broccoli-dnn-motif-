import numpy as np
import pandas as pd
import torch.optim as optim
from sklearn import metrics
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring
from torch.utils.data import DataLoader, Dataset

# from models import *
from modelsv2 import *


# utils

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


def one_hot_e_column(column: pd.Series) -> np.ndarray:
  tmp_list: list = [one_hot_e(seq) for seq in column]
  encoded_column = np.asarray(tmp_list)
  return encoded_column


def reverse_dna_seq(dna_seq: str) -> str:
  # m_reversed = ""
  # for i in range(0, len(dna_seq)):
  #     m_reversed = dna_seq[i] + m_reversed
  # return m_reversed
  return dna_seq[::-1]


def complement_dna_seq(dna_seq: str) -> str:
  comp_map = {"A": "T", "C": "G", "T": "A", "G": "C",
              "a": "t", "c": "g", "t": "a", "g": "c",
              "N": "N", "H": "H", "-": "-",
              "n": "n", "h": "h"
              }

  comp_dna_seq_list: list = [comp_map[nucleotide] for nucleotide in dna_seq]
  comp_dna_seq: str = "".join(comp_dna_seq_list)
  return comp_dna_seq


def reverse_complement_dna_seq(dna_seq: str) -> str:
  return reverse_dna_seq(complement_dna_seq(dna_seq))


def reverse_complement_dna_seqs(column: pd.Series) -> pd.Series:
  tmp_list: list = [reverse_complement_dna_seq(seq) for seq in column]
  rc_column = pd.Series(tmp_list)
  return rc_column

#
# class CNN1D(nn.Module):
#   def __init__(self,
#                in_channel_num_of_nucleotides=4,
#                kernel_size_k_mer_motif=4,
#                dnn_size=256,
#                num_filters=1,
#                lstm_hidden_size=128,
#                *args, **kwargs):
#     super().__init__(*args, **kwargs)
#     self.conv1d = nn.Conv1d(in_channels=in_channel_num_of_nucleotides, out_channels=num_filters,
#                             kernel_size=kernel_size_k_mer_motif, stride=2)
#     self.activation = nn.ReLU()
#     self.pooling = nn.MaxPool1d(kernel_size=kernel_size_k_mer_motif, stride=2)
#
#     self.flatten = nn.Flatten()
#     # linear layer
#
#     self.dnn2 = nn.Linear(in_features=14 * num_filters, out_features=dnn_size)
#     self.act2 = nn.Sigmoid()
#     self.dropout2 = nn.Dropout(p=0.2)
#
#     self.out = nn.Linear(in_features=dnn_size, out_features=1)
#     self.out_act = nn.Sigmoid()
#
#     pass
#
#   def forward(self, x):
#     timber.debug(constants.magenta + f"h0: {x}")
#     h = self.conv1d(x)
#     timber.debug(constants.green + f"h1: {h}")
#     h = self.activation(h)
#     timber.debug(constants.magenta + f"h2: {h}")
#     h = self.pooling(h)
#     timber.debug(constants.blue + f"h3: {h}")
#     timber.debug(constants.cyan + f"h4: {h}")
#
#     h = self.flatten(h)
#     timber.debug(constants.magenta + f"h5: {h},\n shape {h.shape}, size {h.size}")
#     h = self.dnn2(h)
#     timber.debug(constants.green + f"h6: {h}")
#
#     h = self.act2(h)
#     timber.debug(constants.blue + f"h7: {h}")
#
#     h = self.dropout2(h)
#     timber.debug(constants.cyan + f"h8: {h}")
#
#     h = self.out(h)
#     timber.debug(constants.magenta + f"h9: {h}")
#
#     h = self.out_act(h)
#     timber.debug(constants.green + f"h10: {h}")
#     # h = (h > 0.5).float()  # <---- should this go here?
#     # timber.debug(constants.green + f"h11: {h}")
#
#     return h
#

class CustomDataset(Dataset):
  def __init__(self, dataframe):
    self.x = dataframe["Sequence"]
    self.y = dataframe["class"]

  def __len__(self):
    return len(self.y)

  def preprocessing(self, x1, y1) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    forward_col = x1

    backward_col = reverse_complement_dna_seqs(forward_col)

    forward_one_hot_e_col: np.ndarray = one_hot_e_column(forward_col)
    backward_one_hot_e_col: np.ndarray = one_hot_e_column(backward_col)

    tr_xf_tensor = torch.Tensor(forward_one_hot_e_col).permute(1, 2, 0)
    tr_xb_tensor = torch.Tensor(backward_one_hot_e_col).permute(1, 2, 0)
    # timber.debug(f"y1 {y1}")
    tr_y1 = np.array([y1])  # <--- need to put it inside brackets

    return tr_xf_tensor, tr_xb_tensor, tr_y1

  def __getitem__(self, idx):
    m_seq = self.x.iloc[idx]
    labels = self.y.iloc[idx]
    xf, xb, y = self.preprocessing(m_seq, labels)
    timber.debug(f"xf -> {xf.shape}, xb -> {xb.shape}, y -> {y}")
    return xf, xb, y


def test_dataloader():
  df = pd.read_csv("todo.csv")
  X = df["Sequence"]
  y = df["class"]

  ds = CustomDataset(df)
  loader = DataLoader(ds, shuffle=True, batch_size=16)

  train_loader = loader

  for data in train_loader:
    timber.debug(data)
    # xf, xb, y = data[0], data[1], data[2]
    # timber.debug(f"xf -> {xf.shape}, xb -> {xb.shape}, y -> {y.shape}")
  pass


def get_callbacks() -> list:
  # metric.auc ( uses trapezoidal rule) gave an error: x is neither increasing, nor decreasing. so I had to remove it
  return [
    ("tr_acc", EpochScoring(
      metrics.accuracy_score,
      lower_is_better=False,
      on_train=True,
      name="train_acc",
    )),

    ("tr_recall", EpochScoring(
      metrics.recall_score,
      lower_is_better=False,
      on_train=True,
      name="train_recall",
    )),
    ("tr_precision", EpochScoring(
      metrics.precision_score,
      lower_is_better=False,
      on_train=True,
      name="train_precision",
    )),
    ("tr_roc_auc", EpochScoring(
      metrics.roc_auc_score,
      lower_is_better=False,
      on_train=False,
      name="tr_auc"
    )),
    ("tr_f1", EpochScoring(
      metrics.f1_score,
      lower_is_better=False,
      on_train=False,
      name="tr_f1"
    )),
    # ("valid_acc1", EpochScoring(
    #   metrics.accuracy_score,
    #   lower_is_better=False,
    #   on_train=False,
    #   name="valid_acc1",
    # )),
    ("valid_recall", EpochScoring(
      metrics.recall_score,
      lower_is_better=False,
      on_train=False,
      name="valid_recall",
    )),
    ("valid_precision", EpochScoring(
      metrics.precision_score,
      lower_is_better=False,
      on_train=False,
      name="valid_precision",
    )),
    ("valid_roc_auc", EpochScoring(
      metrics.roc_auc_score,
      lower_is_better=False,
      on_train=False,
      name="valid_auc"
    )),
    ("valid_f1", EpochScoring(
      metrics.f1_score,
      lower_is_better=False,
      on_train=False,
      name="valid_f1"
    ))
  ]


def start():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = CnnLstm1DNoBatchNormV3NoActivation().to(device)  # get_stackoverflow_model().to(device)
  # df = pd.read_csv("data64.csv")  # use this line
  df = pd.read_csv("data2000random.csv")



  X = df["Sequence"]
  y = df["class"]

  npa = np.array([y.values])

  torch_tensor = torch.tensor(npa)  # [0, 1, 1, 0, ... ... ] a simple list
  print(f"torch_tensor: {torch_tensor}")
  # need to transpose it!

  yt = torch.transpose(torch_tensor, 0, 1)

  ds = CustomDataset(df)
  loader = DataLoader(ds, shuffle=True)

  # train_loader = loader
  # test_loader = loader  # todo: load another dataset later




  # model = CnnLstm1DNoBatchNormV2().to(device) # get_stackoverflow_model().to(device)
  m_criterion = nn.BCEWithLogitsLoss
  # optimizer = optim.Adam(model.parameters(), lr=0.001)
  m_optimizer = optim.Adam

  net = NeuralNetClassifier(
    model,
    max_epochs=50,
    criterion=m_criterion,
    optimizer=m_optimizer,
    lr=0.01,
    # decay=0.01,
    # momentum=0.9,

    device=device,
    classes=["no_mqtl", "yes_mqtl"],
    verbose=True,
    callbacks=get_callbacks()
  )

  ohe_c = one_hot_e_column(X)
  print(f"ohe_c shape {ohe_c.shape}")
  ohe_c = torch.Tensor(ohe_c)
  ohe_c = ohe_c.permute(0, 2, 1)
  ohe_c = ohe_c.to(device)
  print(f"ohe_c shape {ohe_c.shape}")

  net.fit(X=ohe_c, y=yt)
  y_proba = net.predict_proba(ohe_c)
  # timber.info(f"y_proba = {y_proba}")
  pass


if __name__ == '__main__':
  start()
  # test_dataloader()
  pass


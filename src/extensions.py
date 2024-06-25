import logging

import numpy as np
import pandas as pd
import torch
from skorch import NeuralNetClassifier
from torch.utils.data import Dataset
from sklearn import metrics
from skorch.callbacks import EpochScoring

timber = logging.getLogger()
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)  # change to level=logging.DEBUG to print more logs...


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
  encoded_transposed = encoded.transpose()  # todo: Needs review
  return encoded_transposed


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
    # ("tr_precision", EpochScoring(
    #   metrics.precision_score,
    #   lower_is_better=False,
    #   on_train=True,
    #   name="train_precision",
    # )),
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
    # ("valid_precision", EpochScoring(
    #   metrics.precision_score,
    #   lower_is_better=False,
    #   on_train=False,
    #   name="valid_precision",
    # )),
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


class MyDataSet(Dataset):
  def __init__(self, X: pd.Series, y: pd.Series):
    self.X = X
    self.y = y
    self.len = len(X)

  def __len__(self):
    return self.len

  def __getitem__(self, idx):
    seq, label = self.X.values[idx], self.y.values[idx]
    seq_rc = reverse_complement_dna_seq(seq)
    ohe_seq = one_hot_e(dna_seq=seq)
    # print(f"shape fafafa = { ohe_seq.shape = }")
    ohe_seq_rc = one_hot_e(dna_seq=seq_rc)

    label_number = label * 1.0
    label_np_array = np.asarray([label_number])
    # return ohe_seq, ohe_seq_rc, label
    return [ohe_seq, ohe_seq_rc], label_np_array

  def __getitem_v1__(self, idx):
    seq, label = self.X.values[idx], self.y.values[idx]
    # seq_rc = reverse_complement_dna_seq(seq)
    ohe_seq = one_hot_e(dna_seq=seq).transpose()  # todo: Needs review
    # print(f"shape fafafa = { ohe_seq.shape = }")
    # ohe_seq_rc = one_hot_e(dna_seq=seq_rc)
    # return ohe_seq, ohe_seq_rc, label
    # return [ohe_seq, ohe_seq_rc], label
    label_number = label
    label_np_array = np.asarray([label_number])
    return ohe_seq, label_np_array


class TrainValidDataset(Dataset):
  def __init__(self, train_ds: Dataset, valid_ds: Dataset):
    self.train_ds = train_ds
    self.valid_ds = valid_ds  # or use it as test dataset
    pass


class EmptyDataset(Dataset):
  def __init__(self):
    pass

  def __len__(self):
    return 0


class TestDataset(TrainValidDataset):
  def __init__(self, test_ds: Dataset):
    super().__init__(train_ds=EmptyDataset(), valid_ds=test_ds)


class MQtlNeuralNetClassifier(NeuralNetClassifier):
  def get_split_datasets(self, X, y=None, **fit_params):
    # overriding this function
    dataset = self.get_dataset(X, y)
    if isinstance(X, TrainValidDataset):
      dataset_train, dataset_valid = X.train_ds, X.valid_ds
      return dataset_train, dataset_valid
    raise AssertionError("X is not a TrainValidDataset!")

import logging

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

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


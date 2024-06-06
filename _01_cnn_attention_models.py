import logging

import torch.nn as nn
import torch
from torch.utils._contextlib import F

import constants

# from models import TimeDistributed, ReduceSumLambdaLayer

timber = logging.getLogger()
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)  # change to level=logging.DEBUG to print more logs...


class CnnAttentionModel(nn.Module):

  def __init__(self,
               in_channel_num_of_nucleotides=4,
               kernel_size_k_mer_motif=8,
               seq_len=8000,
               num_classes=1,
               *args,
               **kwargs
               ):
    super().__init__(*args, **kwargs)

    hidden_size_is_len_of_a_sequence_divided_by_kernel_size = int(seq_len / kernel_size_k_mer_motif)
    # ie, seq_len = 8k, kernel = 8, hence hidden_size = 1k

    self.input_channels = in_channel_num_of_nucleotides
    self.hidden_size = hidden_size_is_len_of_a_sequence_divided_by_kernel_size
    self.num_classes = num_classes

    # define cnn layer
    self.cnn_seq = nn.Sequential(
      nn.Conv1d(in_channels=self.input_channels, out_channels=self.hidden_size, kernel_size=kernel_size_k_mer_motif,
                padding="same"),
      nn.Sigmoid(),
      nn.MaxPool1d(kernel_size=kernel_size_k_mer_motif)
    )

    # self.softmax = nn.Softmax()

    # define attention weights
    self.attention_weights = nn.Linear(self.hidden_size, 1)
    # define output layer
    self.fc = nn.Linear(self.hidden_size, self.num_classes)
    pass

  def forward(self, x):
    # pass through the cnn seq layer
    cnn_output = self.cnn_seq(x)

    # compute the attention scores
    attention_score = self.attention_weights(cnn_output).squeeze(-1)
    m_attention_weights = nn.functional.softmax(attention_score, dim=1)

    # compute the attention context vector
    context_vector = torch.sum(m_attention_weights.unsqueeze(-1) * cnn_output, dim=2)
    # Classify using the context vector
    y = self.fc(context_vector)
    return y

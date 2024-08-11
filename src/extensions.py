import logging

import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset

import traceback
import torch
from captum.attr import IntegratedGradients, DeepLiftShap, DeepLift
from transformers import BertTokenizer, BatchEncoding

import mycolors

timber = logging.getLogger()
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)  # change to level=logging.DEBUG to print more logs...

DNA_BERT_6 = "zhihan1996/DNA_bert_6"

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
  encoded_column = np.asarray(tmp_list).astype(np.float32)
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


def reverse_complement_column(column: pd.Series) -> np.ndarray:
  rc_column: list = [reverse_complement_dna_seq(seq) for seq in column]
  return rc_column


class BERTDataSet(Dataset):
  def __init__(self, X: pd.Series, y: pd.Series):
    self.X = X
    self.y = y
    self.len = len(X)
    dna_bert_name: str = DNA_BERT_6
    self.bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=dna_bert_name)

  def preprocess(self, sequence: str, k=6) -> list[str]:
    tokens = [sequence[i:i + k] for i in range(len(sequence) - k + 1)]
    return tokens

  def __len__(self):
    return self.len

  def __getitem__(self, idx):
    seq, label = self.X.values[idx], self.y.values[idx]
    # timber.debug(red + f"{label = }")
    tokens: list[str] = self.preprocess(seq)
    # timber.debug(green + f"{tokens = }")

    encoded_input_x: BatchEncoding = self.bert_tokenizer(
      tokens, return_tensors='pt', is_split_into_words=True, padding='max_length',
      truncation=True, max_length=512
    )
    # torch.Size([128, 1, 512]) --> [128, 512]
    # torch.Size([16, 1, 512]) --> [16, 512]
    # encoded_input_x_2d = [ (key, value.squeeze(dim=1).to(DEVICE) ) for (key, value) in encoded_input_x_3d.items() ]
    input_ids: torch.tensor = encoded_input_x["input_ids"]
    token_type_ids: torch.tensor = encoded_input_x["token_type_ids"]
    attention_mask: torch.tensor = encoded_input_x["attention_mask"]

    # encoded_input_x["input_ids"] = input_ids.squeeze(dim=1).to(DEVICE)
    # encoded_input_x["token_type_ids"] = token_type_ids.squeeze(dim=1).to(DEVICE)
    # encoded_input_x["attention_mask"] = attention_mask.squeeze(dim=1).to(DEVICE)
    encoded_input_x = {key: val.squeeze() for key, val in encoded_input_x.items()}
    return encoded_input_x, label


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
    label_np_array = np.asarray([label_number]).astype(np.float32)
    # return ohe_seq, ohe_seq_rc, label
    return [ohe_seq, ohe_seq_rc], label_np_array


######################################

# functions for interpretability


def interpret_using_integrated_gradients(pytorch_model, input_tensor: torch.Tensor, output_tensor: torch.Tensor):
  # Integrated Gradients
  timber.info("\n\n-------- Integrated Gradient start --------\n\n")
  input_tensor.requires_grad_()
  # xb_tensor.requires_grad_()
  ig: IntegratedGradients = IntegratedGradients(pytorch_model)

  ig_attr, ig_delta = ig.attribute(input_tensor, return_convergence_delta=True)
  timber.info(f"ig_attr: {ig_attr}\n\n")
  timber.info(f"ig_delta: {ig_delta}\n\n")
  timber.info("\n\n-------- Integrated Gradient end --------\n\n")
  return ig_attr


def interpret_using_deeplift(pytorch_model, input_stacked_tensors: torch.Tensor, output_tensors: torch.Tensor):
  timber.info("\n\n-------- DeepLift start --------\n\n")
  input_stacked_tensors.requires_grad_()
  # xb_tensor.requires_grad_()
  deeplift = DeepLift(pytorch_model)

  dl_attr, dl_delta = deeplift.attribute(input_stacked_tensors, return_convergence_delta=True,
                                         baselines=(input_stacked_tensors * 0))
  timber.info(f"dl_attr: {dl_attr}\n\n")
  timber.info(f"dl_delta: {dl_delta}\n\n")
  timber.info("\n\n-------- DeepLift end --------\n\n")

  return dl_attr


def interpret_using_deeplift_shap(pytorch_model, input_stacked_tensors: torch.Tensor, output_tensors: torch.Tensor):
  timber.info("\n\n-------- DeepLiftShap start --------\n\n")
  input_stacked_tensors.requires_grad_()
  # xb_tensor.requires_grad_()
  deeplift_shap = DeepLiftShap(pytorch_model)
  dls_attr, dls_delta = deeplift_shap.attribute(input_stacked_tensors, return_convergence_delta=True,
                                                baselines=(input_stacked_tensors * 0))
  timber.info(f"dsl_attr: {dls_attr}\n\n")
  timber.info(f"dls_delta: {dls_delta}\n\n")
  timber.info("\n\n-------- DeepLiftShap end --------\n\n")

  return dls_attr


def interpret_model(pytorch_model, xf_tensor: torch.Tensor, xb_tensor: torch.Tensor,
                    output_tensor: torch.Tensor):
  # xf_tensor = xf_tensor.to(device=device)
  # xb_tensor = xb_tensor.to(device=device)
  # output_tensor = output_tensor.to(device=device)
  pytorch_model = pytorch_model.to(device="cpu")
  # ensure evaluation mode
  pytorch_model = pytorch_model.eval()

  try:
    interpret_using_integrated_gradients(pytorch_model, xf_tensor, xb_tensor, output_tensor)
  except Exception as x:
    timber.error(mycolors.red + f"interpret_using_integrated_gradients: {x}")
    timber.error(mycolors.yellow + traceback.format_exc())

  try:
    interpret_using_deeplift(pytorch_model, xf_tensor, xb_tensor, output_tensor)
  except Exception as x:
    timber.error(mycolors.red + f"interpret_using_deeplift: {x}")
    timber.error(mycolors.yellow + traceback.format_exc())
  try:
    interpret_using_deeplift_shap(pytorch_model, xf_tensor, xb_tensor, output_tensor)
  except Exception as x:
    timber.error(mycolors.red + f"interpret_using_deeplift_shap: {x}")
    timber.error(mycolors.yellow + traceback.format_exc())
  pass


# Some more util functions!
def create_conv_sequence(in_channel_num_of_nucleotides, num_filters, kernel_size_k_mer_motif) -> nn.Sequential:
  conv1d = nn.Conv1d(in_channels=in_channel_num_of_nucleotides, out_channels=num_filters,
                     kernel_size=kernel_size_k_mer_motif,
                     padding="same")  # stride = 2, just dont use stride, keep it simple for now
  activation = nn.ReLU(inplace=False)  # (inplace=True) will fess with interpretability
  pooling = nn.MaxPool1d(
    kernel_size=kernel_size_k_mer_motif)  # stride = 2, just dont use stride, keep it simple for now

  return nn.Sequential(conv1d, activation, pooling)


class LambdaLayer(nn.Module):
  """
    tensorflow to pytorch lambda layer: https://discuss.pytorch.org/t/how-to-implement-keras-layers-core-lambda-in-pytorch/5903/2?u=fahimfarhan
    """

  def __init__(self, m_lambda):
    super(LambdaLayer, self).__init__()
    self.m_lambda = m_lambda

  def forward(self, x):
    return self.m_lambda(x)


class ReduceSumLambdaLayer(LambdaLayer):
  def __init__(self, m_lambda=torch.sum, m_dim=2):
    super().__init__(m_lambda)
    self.dim = m_dim  # in tensorflow, this dim was 1 :/

  def forward(self, x):
    return self.m_lambda(input=x, dim=self.dim)  # torch.sum(x,dim= 1)


class TimeDistributed(nn.Module):
  def __init__(self, module):
    super(TimeDistributed, self).__init__()
    self.module = module

  def forward(self, x):
    if len(x.size()) <= 2:
      return self.module(x)
    t, n = x.size(0), x.size(1)
    # merge batch and seq dimensions
    x_reshape = x.contiguous().view(t * n, x.size(2))
    y = self.module(x_reshape)
    # We have to reshape Y
    y = y.contiguous().view(t, n, y.size()[1])
    return y


class CommonAttentionLayer(nn.Module):
  def __init__(self, hidden_size, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.attention_linear = nn.Linear(hidden_size, 1)
    pass

  def forward(self, hidden_states):
    # Apply linear layer
    attn_weights = self.attention_linear(hidden_states)
    # Apply softmax to get attention scores
    attn_weights = torch.softmax(attn_weights, dim=1)
    # Apply attention weights to hidden states
    context_vector = torch.sum(attn_weights * hidden_states, dim=1)
    return context_vector, attn_weights


# Function to convert a PyTorch model to a dictionary
def model_to_dict(model):
  model_dict = {}
  for name, module in model.named_modules():
    if len(list(module.children())) == 0:  # Check if the module has no children
      model_dict[name] = {k: v for k, v in module.state_dict().items()}
  return model_dict

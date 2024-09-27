import logging
import os
import random
from typing import Any

import numpy as np
import pandas as pd
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from pytorch_lightning.utilities.types import OptimizerLRScheduler, STEP_OUTPUT, EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryRecall
from transformers import BertModel, BatchEncoding, BertTokenizer, TrainingArguments
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
import torch
from torch import nn
from datasets import load_dataset, IterableDataset
from huggingface_hub import PyTorchModelHubMixin

from dotenv import load_dotenv
from huggingface_hub import login

timber = logging.getLogger()
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)  # change to level=logging.DEBUG to print more logs...

black = "\u001b[30m"
red = "\u001b[31m"
green = "\u001b[32m"
yellow = "\u001b[33m"
blue = "\u001b[34m"
magenta = "\u001b[35m"
cyan = "\u001b[36m"
white = "\u001b[37m"

FORWARD = "FORWARD_INPUT"
BACKWARD = "BACKWARD_INPUT"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def login_inside_huggingface_virtualmachine():
  # Load the .env file, but don't crash if it's not found (e.g., in Hugging Face Space)
  try:
    load_dotenv()  # Only useful on your laptop if .env exists
    print(".env file loaded successfully.")
  except Exception as e:
    print(f"Warning: Could not load .env file. Exception: {e}")

  # Try to get the token from environment variables
  try:
    token = os.getenv("HF_TOKEN")

    if not token:
      raise ValueError("HF_TOKEN not found. Make sure to set it in the environment variables or .env file.")

    # Log in to Hugging Face Hub
    login(token)
    print("Logged in to Hugging Face Hub successfully.")

  except Exception as e:
    print(f"Error during Hugging Face login: {e}")
    # Handle the error appropriately (e.g., exit or retry)


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


class TorchMetrics:
  def __init__(self, device=DEVICE):
    self.binary_accuracy = BinaryAccuracy().to(device)
    self.binary_auc = BinaryAUROC().to(device)
    self.binary_f1_score = BinaryF1Score().to(device)
    self.binary_precision = BinaryPrecision().to(device)
    self.binary_recall = BinaryRecall().to(device)
    pass

  def update_on_each_step(self, batch_predicted_labels, batch_actual_labels):  # todo: Add log if needed
    self.binary_accuracy.update(preds=batch_predicted_labels, target=batch_actual_labels)
    self.binary_auc.update(preds=batch_predicted_labels, target=batch_actual_labels)
    self.binary_f1_score.update(preds=batch_predicted_labels, target=batch_actual_labels)
    self.binary_precision.update(preds=batch_predicted_labels, target=batch_actual_labels)
    self.binary_recall.update(preds=batch_predicted_labels, target=batch_actual_labels)
    pass

  def compute_and_reset_on_epoch_end(self, log, log_prefix: str, log_color: str = green):
    b_accuracy = self.binary_accuracy.compute()
    b_auc = self.binary_auc.compute()
    b_f1_score = self.binary_f1_score.compute()
    b_precision = self.binary_precision.compute()
    b_recall = self.binary_recall.compute()
    timber.info(
      log_color + f"{log_prefix}_acc = {b_accuracy}, {log_prefix}_auc = {b_auc}, {log_prefix}_f1_score = {b_f1_score}, {log_prefix}_precision = {b_precision}, {log_prefix}_recall = {b_recall}")
    log(f"{log_prefix}_accuracy", b_accuracy)
    log(f"{log_prefix}_auc", b_auc)
    log(f"{log_prefix}_f1_score", b_f1_score)
    log(f"{log_prefix}_precision", b_precision)
    log(f"{log_prefix}_recall", b_recall)

    self.binary_accuracy.reset()
    self.binary_auc.reset()
    self.binary_f1_score.reset()
    self.binary_precision.reset()
    self.binary_recall.reset()
    pass


def insert_debug_motif_at_random_position(seq, DEBUG_MOTIF):
  start = 0
  end = len(seq)
  rand_pos = random.randrange(start, (end - len(DEBUG_MOTIF)))
  random_end = rand_pos + len(DEBUG_MOTIF)
  output = seq[start: rand_pos] + DEBUG_MOTIF + seq[random_end: end]
  assert len(seq) == len(output)
  return output


class PagingMQTLDataset(IterableDataset):
  def __init__(self, m_dataset, seq_len, check_if_pipeline_is_ok_by_inserting_debug_motif=False):
    self.dataset = m_dataset
    self.check_if_pipeline_is_ok_by_inserting_debug_motif = check_if_pipeline_is_ok_by_inserting_debug_motif
    self.debug_motif = "ATCGCCTA"
    self.seq_len = seq_len
    pass

  def __iter__(self):
    for row in self.dataset:
      processed = self.preprocess(row)
      if processed is not None:
        yield processed

  def preprocess(self, row):
    seq = row['sequence']  # Fetch the 'sequence' column
    if len(seq) != self.seq_len:
      return None  # skip problematic row!
    label = row['label']  # Fetch the 'label' column (or whatever target you use)
    if label == 1 and self.check_if_pipeline_is_ok_by_inserting_debug_motif:
      seq = insert_debug_motif_at_random_position(seq=seq, DEBUG_MOTIF=self.debug_motif)
    #   todo: perform the preprocessing
    """
    seq_rc = reverse_complement_dna_seq(seq)
    ohe_seq = one_hot_e(dna_seq=seq)
    # print(f"shape fafafa = { ohe_seq.shape = }")
    ohe_seq_rc = one_hot_e(dna_seq=seq_rc)

    label_number = label * 1.0
    label_np_array = np.asarray([label_number]).astype(np.float32)
    # return ohe_seq, ohe_seq_rc, label
    return [ohe_seq, ohe_seq_rc], label_np_array
    """
    return None



# def collate_fn(batch):
#   sequences, labels = zip(*batch)
#   ohe_seq, ohe_seq_rc = sequences[0], sequences[1]
#   # Pad sequences to the maximum length in this batch
#   padded_sequences = pad_sequence(ohe_seq, batch_first=True, padding_value=0)
#   padded_sequences_rc = pad_sequence(ohe_seq_rc, batch_first=True, padding_value=0)
#   # Convert labels to a tensor
#   labels = torch.stack(labels)
#   return [padded_sequences, padded_sequences_rc], labels


class MqtlDataModule(LightningDataModule):
  def __init__(self, train_ds, val_ds, test_ds, batch_size=16):
    super().__init__()
    self.batch_size = batch_size
    self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False,
                                   # collate_fn=collate_fn,
                                   num_workers=1,
                                   # persistent_workers=True
                                   )
    self.validate_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False,
                                      # collate_fn=collate_fn,
                                      num_workers=1,
                                      # persistent_workers=True
                                      )
    self.test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False,
                                  # collate_fn=collate_fn,
                                  num_workers=1,
                                  # persistent_workers=True
                                  )
    pass

  def prepare_data(self):
    pass

  def setup(self, stage: str) -> None:
    timber.info(f"inside setup: {stage = }")
    pass

  def train_dataloader(self) -> TRAIN_DATALOADERS:
    return self.train_loader

  def val_dataloader(self) -> EVAL_DATALOADERS:
    return self.validate_loader

  def test_dataloader(self) -> EVAL_DATALOADERS:
    return self.test_loader


class MQtlClassifierLightningModule(LightningModule):
  def __init__(self,
               classifier: nn.Module,
               criterion=nn.BCELoss(),  # nn.BCEWithLogitsLoss(),
               regularization: int = 2,  # 1 == L1, 2 == L2, 3 (== 1 | 2) == both l1 and l2, else ignore / don't care
               l1_lambda=0.001,
               l2_wright_decay=0.001,
               m_optimizer=torch.optim.Adam,
               *args: Any,
               **kwargs: Any):
    super().__init__(*args, **kwargs)
    self.classifier = classifier
    self.criterion = criterion
    self.train_metrics = TorchMetrics()
    self.validate_metrics = TorchMetrics()
    self.test_metrics = TorchMetrics()

    self.regularization = regularization
    self.l1_lambda = l1_lambda
    self.l2_weight_decay = l2_wright_decay
    self.m_optimizer = m_optimizer
    pass

  def forward(self, x, *args: Any, **kwargs: Any) -> Any:
    return self.classifier.forward(x)

  def configure_optimizers(self) -> OptimizerLRScheduler:
    # Here we add weight decay (L2 regularization) to the optimizer
    weight_decay = 0.0
    if self.regularization == 2 or self.regularization == 3:
      weight_decay = self.l2_weight_decay
    return self.m_optimizer(self.parameters(), lr=1e-3, weight_decay=weight_decay)  # , weight_decay=0.005)

  def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
    # Accuracy on training batch data
    x, y = batch
    x = [i.float() for i in x]
    preds = self.forward(x)
    loss = self.criterion(preds, y)

    if self.regularization == 1 or self.regularization == 3:  # apply l1 regularization
      l1_norm = sum(p.abs().sum() for p in self.parameters())
      loss += self.l1_lambda * l1_norm

    self.log("train_loss", loss)
    # calculate the scores start
    self.train_metrics.update_on_each_step(batch_predicted_labels=preds, batch_actual_labels=y)
    # calculate the scores end
    return loss

  def on_train_epoch_end(self) -> None:
    timber.info(green + "on_train_epoch_end")
    self.train_metrics.compute_and_reset_on_epoch_end(log=self.log, log_prefix="train")
    pass

  def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
    # Accuracy on validation batch data
    x, y = batch
    x = [i.float() for i in x]

    preds = self.forward(x)
    loss = self.criterion(preds, y)
    self.log("valid_loss", loss)
    # calculate the scores start
    self.validate_metrics.update_on_each_step(batch_predicted_labels=preds, batch_actual_labels=y)
    # calculate the scores end
    return loss

  def on_validation_epoch_end(self) -> None:
    timber.info(blue + "on_validation_epoch_end")
    self.validate_metrics.compute_and_reset_on_epoch_end(log=self.log, log_prefix="validate", log_color=blue)
    return None

  def test_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
    # Accuracy on validation batch data
    x, y = batch
    x = [i.float() for i in x]

    preds = self.forward(x)
    loss = self.criterion(preds, y)
    self.log("test_loss", loss)  # do we need this?
    # calculate the scores start
    self.test_metrics.update_on_each_step(batch_predicted_labels=preds, batch_actual_labels=y)
    # calculate the scores end
    return loss

  def on_test_epoch_end(self) -> None:
    timber.info(magenta + "on_test_epoch_end")
    self.test_metrics.compute_and_reset_on_epoch_end(log=self.log, log_prefix="test", log_color=magenta)
    return None

  pass


def start(classifier_model, model_save_path,  m_optimizer=torch.optim.Adam, WINDOW=200,
         is_binned=True, is_debug=False, max_epochs=10):
  file_suffix = ""
  if is_binned:
    file_suffix = "_binned"

  data_files = {
    # small samples
    "train_binned_200": "/home/soumic/Codes/mqtl-classification/src/inputdata/dataset_200_train_binned.csv",
    "validate_binned_200": "/home/soumic/Codes/mqtl-classification/src/inputdata/dataset_200_validate_binned.csv",
    "test_binned_200": "/home/soumic/Codes/mqtl-classification/src/inputdata/dataset_200_test_binned.csv",
    # large samples
    "train_binned_4000": "/home/soumic/Codes/mqtl-classification/src/inputdata/dataset_4000_train_binned.csv",
    "validate_binned_4000": "/home/soumic/Codes/mqtl-classification/src/inputdata/dataset_4000_validate_binned.csv",
    "test_binned_4000": "/home/soumic/Codes/mqtl-classification/src/inputdata/dataset_4000_test_binned.csv",
  }

  dataset_map = None
  is_my_laptop = os.path.isfile("/home/soumic/Codes/mqtl-classification/src/inputdata/dataset_4000_test_binned.csv")
  if is_my_laptop:
    dataset_map = load_dataset("csv", data_files=data_files, streaming=True)
  else:
    dataset_map = load_dataset("fahimfarhan/mqtl-classification-datasets", streaming=True)

  train_dataset = PagingMQTLDataset(dataset_map[f"train_binned_{WINDOW}"],
                                    check_if_pipeline_is_ok_by_inserting_debug_motif=is_debug,
                                    seq_len=WINDOW
                                    )
  val_dataset = PagingMQTLDataset(dataset_map[f"validate_binned_{WINDOW}"],
                                  check_if_pipeline_is_ok_by_inserting_debug_motif=is_debug,
                                  seq_len=WINDOW)
  test_dataset = PagingMQTLDataset(dataset_map[f"test_binned_{WINDOW}"],
                                   check_if_pipeline_is_ok_by_inserting_debug_motif=is_debug,
                                   seq_len=WINDOW)

  data_module = MqtlDataModule(train_ds=train_dataset, val_ds=val_dataset, test_ds=test_dataset)

  classifier_model = classifier_model  #.to(DEVICE)
  try:
    classifier_model = classifier_model.from_pretrained(classifier_model.model_repository_name)
  except Exception as x:
    print(x)

  classifier_module = MQtlClassifierLightningModule(classifier=classifier_model, regularization=2,
                                                    m_optimizer=m_optimizer)

  # if os.path.exists(model_save_path):
  #   classifier_module.load_state_dict(torch.load(model_save_path))

  classifier_module = classifier_module  # .double()

  trainer = Trainer(max_epochs=max_epochs, precision="32")
  trainer.fit(model=classifier_module, datamodule=data_module)
  timber.info("\n\n")
  trainer.test(model=classifier_module, datamodule=data_module)
  timber.info("\n\n")
  # torch.save(classifier_module.state_dict(), model_save_path)  # deprecated, use classifier_model.save_pretrained(model_subdirectory) instead

  #  save locally
  model_subdirectory = classifier_model.model_repository_name
  classifier_model.save_pretrained(model_subdirectory)

  # push to the hub
  commit_message = f":tada: Push model for window size {WINDOW} from huggingface space"
  if is_my_laptop:
    commit_message = f":tada: Push model for window size {WINDOW} from zephyrus"

  classifier_model.push_to_hub(
    repo_id=f"fahimfarhan/{classifier_model.model_repository_name}",
    # subfolder=f"my-awesome-model-{WINDOW}", subfolder didn't work :/
    commit_message=commit_message  # f":tada: Push model for window size {WINDOW}"
  )

  # reload
  # classifier_model = classifier_model.from_pretrained(f"fahimfarhan/{classifier_model.model_repository_name}")
  # classifier_model = classifier_model.from_pretrained(model_subdirectory)

  pass


class ModelTemplate(nn.Module, PyTorchModelHubMixin):
  def __init__(self, seq_len: int, save_model_in_folder_name: str):
    self.seq_len = seq_len
    self.model_repository_name = save_model_in_folder_name
    pass


if __name__ == '__main__':
  login_inside_huggingface_virtualmachine()

  WINDOW = 4000
  some_model = ModelTemplate(seq_len=WINDOW, save_model_in_folder_name="some_fancy_name")

  start(classifier_model=some_model, model_save_path=some_model.model_repository_name, WINDOW=WINDOW,
        is_debug=True, max_epochs=3)
  pass

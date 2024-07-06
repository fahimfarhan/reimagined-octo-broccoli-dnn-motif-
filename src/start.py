import os
from typing import Any

import pandas as pd
import random

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, OptimizerLRScheduler, STEP_OUTPUT

from extensions import *
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch
import mycolors

# df = pd.read_csv("small_dataset.csv")
WINDOW = 4000
DEBUG_MOTIF = "ATCGTTCA"
# LEN_DEBUG_MOTIF = 8
DEBUG = True


def resize_and_insert_motif_if_debug(seq: str, label: int) -> str:
  # else label is 1
  mid = int(len(seq) / 2)
  start = mid - int(WINDOW / 2)
  end = start + WINDOW

  if label == 0:
    return seq[start: end]

  if not DEBUG:
    return seq[start: end]

  rand_pos = random.randrange(start, (end - len(DEBUG_MOTIF)))
  random_end = rand_pos + len(DEBUG_MOTIF)
  output = seq[start: rand_pos] + DEBUG_MOTIF + seq[random_end: end]
  # print(f"{start = }, { rand_pos = }, { random_end = }, { end = }, { len(DEBUG_MOTIF) = }")
  assert len(output) == WINDOW
  return output


def get_dataframe() -> pd.DataFrame:
  df = pd.read_csv("small_dataset.csv")
  tmp = [resize_and_insert_motif_if_debug(seq=df["sequence"][idx], label=int(df["yes_mqtl"][idx])) for idx in
         df.index]  # todo fix this
  # timber.debug(tmp)
  df["sequence"] = tmp
  shuffle_df = df.sample(frac=1)  # shuffle the dataframe

  return shuffle_df


class MqtlDataModule(LightningDataModule):
  def __init__(self, train_ds: MyDataSet, val_ds: MyDataSet, test_ds: MyDataSet):
    super().__init__()
    self.train_loader = DataLoader(train_ds)
    self.validate_loader = DataLoader(val_ds)
    self.test_loader = DataLoader(test_ds)
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


class SimpleCNN1DmQtlClassifierModule(LightningModule):
  def __init__(self,
               seq_len,
               in_channel_num_of_nucleotides=4,
               kernel_size_k_mer_motif=4,
               num_filters=32,
               lstm_hidden_size=128,
               dnn_size=512,
               criterion = nn.BCEWithLogitsLoss(),
               *args: Any,
               **kwargs: Any):
    super().__init__(*args, **kwargs)
    self.criterion = criterion

    self.seq_layer_forward = self.create_conv_sequence(in_channel_num_of_nucleotides, num_filters,
                                                       kernel_size_k_mer_motif)
    self.seq_layer_backward = self.create_conv_sequence(in_channel_num_of_nucleotides, num_filters,
                                                        kernel_size_k_mer_motif)

    tmp = num_filters  # * in_channel_num_of_nucleotides
    tmp_num_filters = num_filters
    # size = seq_len * 2
    self.conv_seq_0 = self.create_conv_sequence(tmp, tmp_num_filters, kernel_size_k_mer_motif)  # output_size0 = size / kernel_size_k_mer_motif
    # self.conv_seq_1 = self.create_conv_sequence(tmp, tmp_num_filters, kernel_size_k_mer_motif)  # output_size1 = output_size0 / kernel_size_k_mer_motif
    # self.conv_seq_2 = self.create_conv_sequence(tmp, tmp_num_filters, kernel_size_k_mer_motif)  # output_size2 = output_size1 / kernel_size_k_mer_motif

    self.flatten = nn.Flatten()

    # dnn_in_features = num_filters * int(seq_len / kernel_size_k_mer_motif) * 2
    dnn_in_features = num_filters * int(seq_len / kernel_size_k_mer_motif / 2)  # no idea why
    # two because forward_sequence,and backward_sequence

    # dnn_in_features = num_filters * int(seq_len / (kernel_size_k_mer_motif ** 4)) * 2
    # dnn_in_features = num_filters * int(seq_len / (kernel_size_k_mer_motif ** 2)) * 2
    self.dnn = nn.Linear(in_features=dnn_in_features, out_features=dnn_size)
    self.dnn_act = nn.ReLU(inplace=True)
    self.dropout = nn.Dropout(p=0.33)

    self.out = nn.Linear(in_features=dnn_size, out_features=1)
    # self.sigmoid = torch.sigmoid # not needed if using nn.BCEWithLogitsLoss()
    pass

  def create_conv_sequence(self, in_channel_num_of_nucleotides, num_filters, kernel_size_k_mer_motif) -> nn.Sequential:
    conv1d = nn.Conv1d(in_channels=in_channel_num_of_nucleotides, out_channels=num_filters,
                       kernel_size=kernel_size_k_mer_motif,
                       padding="same")  # stride = 2, just dont use stride, keep it simple for now
    activation = nn.ReLU(inplace=True)
    pooling = nn.MaxPool1d(
      kernel_size=kernel_size_k_mer_motif)  # stride = 2, just dont use stride, keep it simple for now

    return nn.Sequential(conv1d, activation, pooling)

  def configure_optimizers(self) -> OptimizerLRScheduler:
    return torch.optim.Adam(self.parameters(), lr=1e-3)

  def forward(self, x, *args: Any, **kwargs: Any) -> Any:
    xf, xb = x[0], x[1]

    hf = self.seq_layer_forward(xf)
    timber.debug(mycolors.red + f"1{ hf.shape = }")
    hb = self.seq_layer_backward(xb)
    timber.debug(mycolors.green + f"2{ hb.shape = }")

    h = torch.concatenate(tensors=(hf, hb), dim=2)
    timber.debug(mycolors.yellow + f"4{ h.shape = } concat")

    # todo: use more / less layers and see what happens
    h = self.conv_seq_0(h)
    timber.debug(mycolors.blue + f"4{ h.shape = } conv_seq_0")

    # h = self.conv_seq_1(h)
    # timber.debug(mycolors.magenta + f"4{ h.shape = } conv_seq_1")

    # h = self.conv_seq_2(h)
    # timber.debug(mycolors.magenta + f"4{ h.shape = } conv_seq_2")

    # h = self.conv1d(xf)
    # timber.debug(mycolors.magenta + f"1{ xf.shape = }")
    # h = self.conv1d(xf)
    # timber.debug(mycolors.magenta + f"2{ h.shape = }")
    # h = self.activation(h)
    # timber.debug(mycolors.magenta + f"3{ h.shape = }")
    # h = self.pooling(h)
    # timber.debug(mycolors.magenta + f"4{ h.shape = }")
    h = self.flatten(h)
    timber.debug(mycolors.magenta + f"5{ h.shape = }")
    h = self.dnn(h)
    timber.debug(mycolors.cyan + f"6{ h.shape = }")
    h = self.dnn_act(h)
    timber.debug(mycolors.red + f"7{ h.shape = }")
    h = self.dropout(h)
    timber.debug(mycolors.green + f"8{ h.shape = }")
    h = self.out(h)
    timber.debug(mycolors.yellow + f"9{ h.shape = }")
    # h = self.sigmoid(h)
    # timber.debug(mycolors.magenta + f"10{ h.shape = }")
    # a sigmoid is already added in the BCEWithLogitsLoss Function. Hence don't use another sigmoid!
    y = h
    return y

  def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
    x, y = batch
    logits = self(x)
    loss = self.criterion(logits, y)
    self.log("train_loss", loss)
    return loss

  def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
    x, y = batch
    logits = self(x)
    loss = self.criterion(logits, y)
    self.log("valid_loss", loss)
    return loss

  def on_validation_epoch_end(self) -> None:
    timber.info("on_validation_epoch_end")
    return None
  pass


def start():
  df: pd.DataFrame = get_dataframe()
  for seq in df["sequence"]:
    # print(f"{len(seq)}")
    assert (len(seq) == WINDOW)

  experiment = 'tutorial_3'
  if not os.path.exists(experiment):
    os.makedirs(experiment)

  x_train, x_tmp, y_train, y_tmp = train_test_split(df["sequence"], df["yes_mqtl"], test_size=0.2)
  x_test, x_val, y_test, y_val = train_test_split(x_tmp, y_tmp, test_size=0.5)

  train_dataset = MyDataSet(x_train, y_train)
  val_dataset = MyDataSet(x_val, y_val)
  test_dataset = MyDataSet(x_test, y_test)

  data_module = MqtlDataModule(train_ds=train_dataset, val_ds=val_dataset, test_ds=test_dataset)
  classifier_module = SimpleCNN1DmQtlClassifierModule(seq_len=WINDOW)
  classifier_module = classifier_module.double()

  trainer = Trainer(max_epochs=10)
  trainer.fit(classifier_module, datamodule=data_module)
  pass


if __name__ == '__main__':
  start()
  pass

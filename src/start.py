import random
from typing import Any
import os
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, OptimizerLRScheduler, STEP_OUTPUT
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryRecall
import viz_sequence
import mycolors
from extensions import *
import grelu.visualize

# df = pd.read_csv("small_dataset.csv")
WINDOW = 100
DEBUG_MOTIF = "ATCGTTCA"
# LEN_DEBUG_MOTIF = 8
DEBUG = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" too much code. Better use skotch to ensure the lightning ai scores are actually correct"""


def sklearn_metrics(name, log, y_true, y_pred):
  # return
  # Compute metrics
  binary_accuracy = accuracy_score(y_true, y_pred.round())
  binary_auc = roc_auc_score(y_true, y_pred)
  binary_f1_score = f1_score(y_true, y_pred.round())
  # binary_precision = precision_score(y_true, y_pred.round())  # usually gets error :/
  binary_recall = recall_score(y_true, y_pred.round())
  # Log metrics using sklearn
  log(f'{name}_accuracy', binary_accuracy)
  log(f'{name}_auc', binary_auc)
  log(f'{name}_f1_score', binary_f1_score)
  # log(f'{name}_precision', binary_precision)
  log(f'{name}_recall', binary_recall)


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


def get_dataframe(shuffled: bool = True) -> pd.DataFrame:
  df = pd.read_csv("small_dataset.csv")
  tmp = [resize_and_insert_motif_if_debug(seq=df["sequence"][idx], label=int(df["yes_mqtl"][idx])) for idx in
         df.index]  # todo fix this
  # timber.debug(tmp)
  df["sequence"] = tmp
  if not shuffled:
    return df
  shuffle_df = df.sample(frac=1)  # shuffle the dataframe

  return shuffle_df


class MqtlDataModule(LightningDataModule):
  def __init__(self, train_ds: MyDataSet, val_ds: MyDataSet, test_ds: MyDataSet, batch_size=16):
    super().__init__()
    self.batch_size = batch_size
    self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=15,
                                   persistent_workers=True)
    self.validate_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=15,
                                      persistent_workers=True)
    self.test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, num_workers=15,
                                  persistent_workers=True)
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

  def compute_and_reset_on_epoch_end(self, log, log_prefix: str, log_color: str = mycolors.green):
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


class MQtlClassifierLightningModule(LightningModule):
  def __init__(self,
               classifier: nn.Module,
               criterion=nn.BCELoss(),  # nn.BCEWithLogitsLoss(),
               regularization: int = 2,  # 1 == L1, 2 == L2, 3 (== 1 | 2) == both l1 and l2, else ignore / don't care
               l1_lambda=0.001,
               l2_wright_decay=0.001,
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
    pass

  def forward(self, x, *args: Any, **kwargs: Any) -> Any:
    return self.classifier.forward(x)

  def configure_optimizers(self) -> OptimizerLRScheduler:
    # Here we add weight decay (L2 regularization) to the optimizer
    weight_decay = 0.0
    if self.regularization == 2 or self.regularization == 3:
      weight_decay = self.l2_weight_decay
    return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=weight_decay)  # , weight_decay=0.005)

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
    timber.info(mycolors.green + "on_train_epoch_end")
    self.train_metrics.compute_and_reset_on_epoch_end(log=self.log, log_prefix="train")
    pass

  def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
    # Accuracy on validation batch data
    x, y = batch
    x = [i.float() for i in x]

    preds = self.forward(x)
    loss = 0  # self.criterion(preds, y)
    self.log("valid_loss", loss)
    # calculate the scores start
    self.validate_metrics.update_on_each_step(batch_predicted_labels=preds, batch_actual_labels=y)
    # calculate the scores end
    return loss

  def on_validation_epoch_end(self) -> None:
    timber.info(mycolors.blue + "on_validation_epoch_end")
    self.validate_metrics.compute_and_reset_on_epoch_end(log=self.log, log_prefix="validate", log_color=mycolors.blue)
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
    timber.info(mycolors.magenta + "on_test_epoch_end")
    self.test_metrics.compute_and_reset_on_epoch_end(log=self.log, log_prefix="test", log_color=mycolors.magenta)
    return None

  pass


def start(classifier_model, model_save_path):
  df: pd.DataFrame = get_dataframe()
  for seq in df["sequence"]:
    # print(f"{len(seq)}")
    assert (len(seq) == WINDOW)

  # experiment = 'tutorial_3'
  # if not os.path.exists(experiment):
  #   os.makedirs(experiment)

  x_train, x_tmp, y_train, y_tmp = train_test_split(df["sequence"], df["yes_mqtl"], test_size=0.2)
  x_test, x_val, y_test, y_val = train_test_split(x_tmp, y_tmp, test_size=0.5)

  train_dataset = MyDataSet(x_train, y_train)
  val_dataset = MyDataSet(x_val, y_val)
  test_dataset = MyDataSet(x_test, y_test)

  data_module = MqtlDataModule(train_ds=train_dataset, val_ds=val_dataset, test_ds=test_dataset)

  classifier_model = classifier_model.to(DEVICE)

  classifier_module = MQtlClassifierLightningModule(classifier=classifier_model, regularization=2)

  if os.path.exists(model_save_path):
    classifier_module.load_state_dict(torch.load(model_save_path))

  classifier_module = classifier_module  # .double()

  trainer = Trainer(max_epochs=10, precision="32")
  trainer.fit(model=classifier_module, datamodule=data_module)
  timber.info("\n\n")
  trainer.test(model=classifier_module, datamodule=data_module)
  timber.info("\n\n")
  torch.save(classifier_module.state_dict(), model_save_path)

  start_interpreting_ig_and_dl(classifier_model)
  start_interpreting_with_dlshap(classifier_model)

  pass


def start_interpreting_ig_and_dl(classifier_model):
  df: pd.DataFrame = get_dataframe(False)

  seq = df.get("sequence")[0: 2]
  print(f" {seq = } ")
  # return
  xf_array = one_hot_e_column(seq)
  rc_column = reverse_complement_column(seq)
  xb_array = one_hot_e_column(rc_column)

  xf_tensor = torch.Tensor(xf_array)
  xb_tensor = torch.Tensor(xb_array)

  stacked_tensors = torch.stack((xf_tensor, xb_tensor))

  ig_tensor = interpret_using_integrated_gradients(classifier_model, stacked_tensors, None)
  ig_score = ig_tensor[0][0].detach().numpy()
  viz_sequence.plot_weights(ig_score, subticks_frequency=int(WINDOW / 10))

  fig_ig = grelu.visualize.plot_tracks(
    ig_score,  # Outputs to plot
    start_pos=0,  # Start coordinate for the x-axis label
    end_pos=None,  # End coordinate for the x-axis label
    titles=["ig"],  # titles for each track
    figsize=(20, 6),  # width, height
  )
  # todo: Maybe save the fig object in a file!
  plt.show()
  ig_score_df = pd.DataFrame(ig_score)
  ig_heatmap = grelu.visualize.plot_ISM(ig_score_df, method="heatmap", figsize=(20, 1.5), center=0)
  plt.show()

  # todo: Debug why logo ain't working!
  # ig_logo = grelu.visualize.plot_ISM(ig_score_df, method="logo", figsize=(20, 1.5), center=0)
  # plt.show()

  # ignore = input("Press any key to continue...")
  dl_tensor = interpret_using_deeplift(classifier_model, stacked_tensors, None)
  dl_score = dl_tensor[0][0].detach().numpy()
  viz_sequence.plot_weights(dl_score, subticks_frequency=int(WINDOW / 10))

  fig_dl = grelu.visualize.plot_tracks(
    dl_score,  # Outputs to plot
    start_pos=0,  # Start coordinate for the x-axis label
    end_pos=None,  # End coordinate for the x-axis label
    titles=["dl"],  # titles for each track
    figsize=(20, 6),  # width, height
  )
  # todo: Maybe save the fig object in a file!
  plt.show()
  dl_score_df = pd.DataFrame(dl_score)

  dl_heatmap = grelu.visualize.plot_ISM(dl_score_df, method="heatmap", figsize=(20, 1.5), center=0)
  plt.show()

  # dl_logo = grelu.visualize.plot_ISM(dl_score_df, method="logo", figsize=(20, 1.5), center=0)
  # plt.show()

  pass


def start_interpreting_with_dlshap(classifier_model):
  df: pd.DataFrame = get_dataframe(False)

  seq = df.get("sequence")[0: 4]  # dlshap needs size 4 -_-
  print(f" {seq = } ")
  # return
  xf_array = one_hot_e_column(seq)
  rc_column = reverse_complement_column(seq)
  xb_array = one_hot_e_column(rc_column)

  xf_tensor = torch.Tensor(xf_array)
  xb_tensor = torch.Tensor(xb_array)

  stacked_tensors = torch.stack((xf_tensor, xb_tensor))
  dl_shap_tensor = interpret_using_deeplift_shap(classifier_model, stacked_tensors, None)
  dl_shap_score = dl_shap_tensor[0][0].detach().numpy()
  viz_sequence.plot_weights(dl_shap_score, subticks_frequency=int(WINDOW / 10))

  fig = grelu.visualize.plot_tracks(
    dl_shap_score,  # Outputs to plot
    start_pos=0,  # Start coordinate for the x-axis label
    end_pos=None,  # End coordinate for the x-axis label
    titles=["dl_shap"],  # titles for each track
    figsize=(20, 6),  # width, height
  )
  # todo: Maybe save the fig object in a file!
  plt.show()
  dl_shap_score_df = pd.DataFrame(dl_shap_score)

  dl_shap_heatmap = grelu.visualize.plot_ISM(dl_shap_score_df, method="heatmap", figsize=(20, 1.5), center=0)
  plt.show()

  # dl_shap_logo = grelu.visualize.plot_ISM(dl_shap_score_df, method="logo", figsize=(20, 1.5), center=0)
  # plt.show()

  pass

"""
if __name__ == '__main__':
  simple_cnn = Cnn1dClassifier(seq_len=WINDOW)
  start(classifier_model=simple_cnn, model_save_path=simple_cnn.file_name)
  pass
"""
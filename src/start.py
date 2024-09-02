import os
import random
from typing import Any

import grelu.interpret.score
import grelu.visualize
from matplotlib import pyplot as plt
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, OptimizerLRScheduler, STEP_OUTPUT
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryRecall

import viz_sequence
from extensions import *

# df = pd.read_csv("small_dataset.csv")
# WINDOW = 200
dataset_folder_prefix = "inputdata/"
DEBUG_MOTIF = "ATCGTTCA"
# LEN_DEBUG_MOTIF = 8
DEBUG = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataframe(WINDOW: int, shuffled: bool = True) -> pd.DataFrame:
  df = pd.read_csv(f"{dataset_folder_prefix}dataset_{WINDOW}_test.csv")
  df = df[df["label"] == 1]
  if not shuffled:
    return df
  shuffle_df = df.sample(frac=1)  # shuffle the dataframe

  return shuffle_df


class MqtlDataModule(LightningDataModule):
  def __init__(self, train_ds: Dataset, val_ds: Dataset, test_ds: Dataset, batch_size=16):
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


class MQtlBertClassifierLightningModule(LightningModule):
  def __init__(self,
               classifier: nn.Module,
               criterion=None,  # nn.BCEWithLogitsLoss(),
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
    input_ids: torch.tensor = x["input_ids"]
    attention_mask: torch.tensor = x["attention_mask"]
    token_type_ids: torch.tensor = x["token_type_ids"]
    # print(f"\n{ type(input_ids) = }, {input_ids = }")
    # print(f"{ type(attention_mask) = }, { attention_mask = }")
    # print(f"{ type(token_type_ids) = }, { token_type_ids = }")

    return self.classifier.forward(input_ids, attention_mask, token_type_ids)

  def configure_optimizers(self) -> OptimizerLRScheduler:
    # Here we add weight decay (L2 regularization) to the optimizer
    weight_decay = 0.0
    if self.regularization == 2 or self.regularization == 3:
      weight_decay = self.l2_weight_decay
    return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=weight_decay)  # , weight_decay=0.005)

  def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
    # Accuracy on training batch data
    x, y = batch
    preds = self.forward(x)
    loss = self.criterion(preds, y)

    if self.regularization == 1 or self.regularization == 3:  # apply l1 regularization
      l1_norm = sum(p.abs().sum() for p in self.parameters())
      loss += self.l1_lambda * l1_norm

    self.log("train_loss", loss)
    # calculate the scores start
    self.train_metrics.update_on_each_step(batch_predicted_labels=preds.squeeze(), batch_actual_labels=y)
    # calculate the scores end
    return loss

  def on_train_epoch_end(self) -> None:
    timber.info(mycolors.green + "on_train_epoch_end")
    self.train_metrics.compute_and_reset_on_epoch_end(log=self.log, log_prefix="train")
    pass

  def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
    # Accuracy on validation batch data
    x, y = batch
    preds = self.forward(x)
    loss = 0  # self.criterion(preds, y)
    self.log("valid_loss", loss)
    # calculate the scores start
    self.validate_metrics.update_on_each_step(batch_predicted_labels=preds.squeeze(), batch_actual_labels=y)
    # calculate the scores end
    return loss

  def on_validation_epoch_end(self) -> None:
    timber.info(mycolors.blue + "on_validation_epoch_end")
    self.validate_metrics.compute_and_reset_on_epoch_end(log=self.log, log_prefix="validate", log_color=mycolors.blue)
    return None

  def test_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
    # Accuracy on validation batch data
    x, y = batch
    preds = self.forward(x)
    loss = self.criterion(preds, y)
    self.log("test_loss", loss)  # do we need this?
    # calculate the scores start
    self.test_metrics.update_on_each_step(batch_predicted_labels=preds.squeeze(), batch_actual_labels=y)
    # calculate the scores end
    return loss

  def on_test_epoch_end(self) -> None:
    timber.info(mycolors.magenta + "on_test_epoch_end")
    self.test_metrics.compute_and_reset_on_epoch_end(log=self.log, log_prefix="test", log_color=mycolors.magenta)
    return None

  pass


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


def start(classifier_model, model_save_path, is_attention_model=False, m_optimizer=torch.optim.Adam, WINDOW=200):
  # experiment = 'tutorial_3'
  # if not os.path.exists(experiment):
  #   os.makedirs(experiment)
  """
  x_train, x_tmp, y_train, y_tmp = train_test_split(df["sequence"], df["label"], test_size=0.2)
  x_test, x_val, y_test, y_val = train_test_split(x_tmp, y_tmp, test_size=0.5)

  train_dataset = MyDataSet(x_train, y_train)
  val_dataset = MyDataSet(x_val, y_val)
  test_dataset = MyDataSet(x_test, y_test)
  """
  train_dataset = MQTLDataSet(file_path=f"{dataset_folder_prefix}dataset_{WINDOW}_train.csv")
  val_dataset = MQTLDataSet(file_path=f"{dataset_folder_prefix}dataset_{WINDOW}_validate.csv")
  test_dataset = MQTLDataSet(file_path=f"{dataset_folder_prefix}dataset_{WINDOW}_test.csv")

  data_module = MqtlDataModule(train_ds=train_dataset, val_ds=val_dataset, test_ds=test_dataset)

  classifier_model = classifier_model.to(DEVICE)

  classifier_module = MQtlClassifierLightningModule(classifier=classifier_model, regularization=2,
                                                    m_optimizer=m_optimizer)

  if os.path.exists(model_save_path):
    classifier_module.load_state_dict(torch.load(model_save_path))

  classifier_module = classifier_module  # .double()

  trainer = Trainer(max_epochs=5, precision="32")
  trainer.fit(model=classifier_module, datamodule=data_module)
  timber.info("\n\n")
  trainer.test(model=classifier_module, datamodule=data_module)
  timber.info("\n\n")
  torch.save(classifier_module.state_dict(), model_save_path)

  start_interpreting_ig_and_dl(classifier_model)
  start_interpreting_with_dlshap(classifier_model)
  # if is_attention_model: # todo: repair it later
  #   start_interpreting_attention_failed(classifier_model)
  pass


def start_bert(classifier_model, model_save_path, criterion, WINDOW=200, batch_size=4):
  train_dataset = BertMQTLDataSet(file_path=f"{dataset_folder_prefix}dataset_{WINDOW}_train.csv")
  val_dataset = BertMQTLDataSet(file_path=f"{dataset_folder_prefix}dataset_{WINDOW}_validate.csv")
  test_dataset = BertMQTLDataSet(file_path=f"{dataset_folder_prefix}dataset_{WINDOW}_test.csv")

  data_module = MqtlDataModule(
    train_ds=train_dataset,
    val_ds=val_dataset,
    test_ds=test_dataset,
    batch_size=batch_size)

  classifier_model = classifier_model.to(DEVICE)

  classifier_module = MQtlBertClassifierLightningModule(
    classifier=classifier_model,
    regularization=2, criterion=criterion)

  if os.path.exists(model_save_path):
    classifier_module.load_state_dict(torch.load(model_save_path))

  classifier_module = classifier_module  # .double()

  trainer = Trainer(max_epochs=10, precision="32")
  trainer.fit(model=classifier_module, datamodule=data_module)
  timber.info("\n\n")
  trainer.test(model=classifier_module, datamodule=data_module)
  timber.info("\n\n")
  torch.save(classifier_module.state_dict(), model_save_path)

  return
  # todo: Repair the interpretation pipeline for bert model
  start_interpreting_ig_and_dl(classifier_model)
  start_interpreting_with_dlshap(classifier_model)

  pass


def start_interpreting_ig_and_dl(classifier_model, WINDOW):
  df: pd.DataFrame = get_dataframe(WINDOW, False)

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

  # ignore = inputdata("Press any key to continue...")
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


def start_interpreting_with_dlshap(classifier_model, WINDOW):
  df: pd.DataFrame = get_dataframe(WINDOW, False)

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


def start_interpreting_attention_failed(classifier_model, WINDOW):

  df: pd.DataFrame = get_dataframe(WINDOW, False)

  seq = df.get("sequence")[0: 4]  # dlshap needs size 4 -_-
  print(f" {seq = } ")
  # return
  xf_array = one_hot_e_column(seq)
  rc_column = reverse_complement_column(seq)
  xb_array = one_hot_e_column(rc_column)

  xf_tensor = torch.Tensor(xf_array)
  xb_tensor = torch.Tensor(xb_array)

  stacked_tensors = torch.stack((xf_tensor, xb_tensor))

  ignore, attention_tensor = classifier_model.forward_for_interpretation(stacked_tensors)

  attention_score = attention_tensor.detach().numpy()
  print(f"{attention_score.shape = }")
  attention_score = np.squeeze(attention_score, axis=2).transpose()
  print(f"{attention_score.shape = }(After squeeze)")
  grelu.visualize.plot_attention_matrix(
    attention_score,
  )
  pass


"""
if __name__ == '__main__':
  simple_cnn = Cnn1dClassifier(seq_len=WINDOW)
  start(classifier_model=simple_cnn, model_save_path=simple_cnn.file_name)
  pass
"""

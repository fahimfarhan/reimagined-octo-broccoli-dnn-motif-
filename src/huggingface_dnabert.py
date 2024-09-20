from typing import Any

from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from pytorch_lightning.utilities.types import OptimizerLRScheduler, STEP_OUTPUT, EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset
from torcheval.metrics import BinaryAccuracy, BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryRecall
from transformers import BertModel, BatchEncoding, BertTokenizer, TrainingArguments
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
import torch
from torch import nn
from datasets import load_dataset

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

DNA_BERT_6 = "zhihan1996/DNA_bert_6"


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


class ReshapedBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
  def forward(self, input, target):
    return super().forward(input.squeeze(), target.float())


class MQtlDnaBERT6Classifier(nn.Module):
  def __init__(self,
               bert_model=BertModel.from_pretrained(pretrained_model_name_or_path=DNA_BERT_6),
               hidden_size=768,
               num_classes=1,
               *args,
               **kwargs
               ):
    super().__init__(*args, **kwargs)

    self.model_name = "MQtlDnaBERT6Classifier"

    self.bert_model = bert_model
    self.attention = CommonAttentionLayer(hidden_size)
    self.classifier = nn.Linear(hidden_size, num_classes)
    pass

  def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, token_type_ids):
    """
    # torch.Size([128, 1, 512]) --> [128, 512]
    input_ids = input_ids.squeeze(dim=1).to(DEVICE)
    # torch.Size([16, 1, 512]) --> [16, 512]
    attention_mask = attention_mask.squeeze(dim=1).to(DEVICE)
    token_type_ids = token_type_ids.squeeze(dim=1).to(DEVICE)
    """
    bert_output: BaseModelOutputWithPoolingAndCrossAttentions = self.bert_model(
      input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids
    )

    last_hidden_state = bert_output.last_hidden_state
    context_vector, ignore_attention_weight = self.attention(last_hidden_state)
    y = self.classifier(context_vector)
    return y


class TorchMetrics:
  def __init__(self):
    self.binary_accuracy = BinaryAccuracy()  #.to(device)
    self.binary_auc = BinaryAUROC()  # .to(device)
    self.binary_f1_score = BinaryF1Score()  # .to(device)
    self.binary_precision = BinaryPrecision()  # .to(device)
    self.binary_recall = BinaryRecall()  # .to(device)
    pass

  def update_on_each_step(self, batch_predicted_labels, batch_actual_labels):  # todo: Add log if needed
    # it looks like the library maintainers changed preds to input, ie, before: preds, now: input
    self.binary_accuracy.update(input=batch_predicted_labels, target=batch_actual_labels)
    self.binary_auc.update(input=batch_predicted_labels, target=batch_actual_labels)
    self.binary_f1_score.update(input=batch_predicted_labels, target=batch_actual_labels)
    self.binary_precision.update(input=batch_predicted_labels, target=batch_actual_labels)
    self.binary_recall.update(input=batch_predicted_labels, target=batch_actual_labels)
    pass

  def compute_and_reset_on_epoch_end(self, log, log_prefix: str, log_color: str = green):
    b_accuracy = self.binary_accuracy.compute()
    b_auc = self.binary_auc.compute()
    b_f1_score = self.binary_f1_score.compute()
    b_precision = self.binary_precision.compute()
    b_recall = self.binary_recall.compute()
    # timber.info(  log_color + f"{log_prefix}_acc = {b_accuracy}, {log_prefix}_auc = {b_auc}, {log_prefix}_f1_score = {b_f1_score}, {log_prefix}_precision = {b_precision}, {log_prefix}_recall = {b_recall}")
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
    self.train_metrics.compute_and_reset_on_epoch_end(log=self.log, log_prefix="train")
    pass

  def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
    # Accuracy on validation batch data
    # print(f"debug { batch = }")
    x, y = batch
    preds = self.forward(x)
    loss = 0  # self.criterion(preds, y)
    self.log("valid_loss", loss)
    # calculate the scores start
    self.validate_metrics.update_on_each_step(batch_predicted_labels=preds.squeeze(), batch_actual_labels=y)
    # calculate the scores end
    return loss

  def on_validation_epoch_end(self) -> None:
    self.validate_metrics.compute_and_reset_on_epoch_end(log=self.log, log_prefix="validate", log_color=blue)
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
    self.test_metrics.compute_and_reset_on_epoch_end(log=self.log, log_prefix="test", log_color=magenta)
    return None

  pass


class DNABERTDataset(Dataset):
  def __init__(self, dataset, tokenizer, max_length=512):
    self.dataset = dataset
    self.bert_tokenizer = tokenizer
    self.max_length = max_length

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    sequence = self.dataset[idx]['sequence']  # Fetch the 'sequence' column
    label = self.dataset[idx]['label']  # Fetch the 'label' column (or whatever target you use)

    # Tokenize the sequence
    encoded_sequence: BatchEncoding = self.bert_tokenizer(
      sequence,
      truncation=True,
      padding='max_length',
      max_length=self.max_length,
      return_tensors='pt'
    )

    encoded_sequence_squeezed = {key: val.squeeze() for key, val in encoded_sequence.items()}
    return encoded_sequence_squeezed, label


class DNABERTDataModule(LightningDataModule):
  def __init__(self, model_name=DNA_BERT_6, batch_size=8):
    super().__init__()
    self.tokenized_dataset = None
    self.dataset = None
    self.train_dataset: DNABERTDataset = None
    self.validate_dataset: DNABERTDataset = None
    self.test_dataset: DNABERTDataset = None
    self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=DNA_BERT_6)
    self.batch_size = batch_size

  def prepare_data(self):
    # Download and prepare dataset
    self.dataset = load_dataset("fahimfarhan/mqtl-classification-dataset-binned-200")

  def setup(self, stage=None):
    self.train_dataset = DNABERTDataset(self.dataset['train'], self.tokenizer)
    self.validate_dataset = DNABERTDataset(self.dataset['validate'], self.tokenizer)
    self.test_dataset = DNABERTDataset(self.dataset['test'], self.tokenizer)

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=15)

  def val_dataloader(self):
    return DataLoader(self.validate_dataset, batch_size=self.batch_size, num_workers=15)

  def test_dataloader(self) -> EVAL_DATALOADERS:
    return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=15)


# Initialize DataModule
model_name = "zhihan1996/DNABERT-6"
data_module = DNABERTDataModule(model_name=model_name, batch_size=8)


def start_bert(classifier_model, model_save_path, criterion, WINDOW=200, batch_size=4,
               dataset_folder_prefix="inputdata/", is_binned=True, is_debug=False, max_epochs=10):
  file_suffix = ""
  if is_binned:
    file_suffix = "_binned"

  data_module = DNABERTDataModule(batch_size=batch_size)

  # classifier_model = classifier_model.to(DEVICE)

  classifier_module = MQtlBertClassifierLightningModule(
    classifier=classifier_model,
    regularization=2, criterion=criterion)

  # if os.path.exists(model_save_path):
  #   classifier_module.load_state_dict(torch.load(model_save_path))

  classifier_module = classifier_module  # .double()

  # Set up training arguments
  training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=max_epochs,
    logging_dir='./logs',
    report_to="none",  # Disable reporting to WandB, etc.
  )

  # Prepare data using the DataModule
  data_module.prepare_data()
  data_module.setup()

  # Initialize Trainer
  # trainer = Trainer(
  #   model=classifier_module,
  #   args=training_args,
  #   train_dataset=data_module.tokenized_dataset["train"],
  #   eval_dataset=data_module.tokenized_dataset["test"],
  # )

  trainer = Trainer(max_epochs=max_epochs, precision="32")

  # Train the model
  trainer.fit(model=classifier_module, datamodule=data_module)
  trainer.test(model=classifier_module, datamodule=data_module)
  torch.save(classifier_module.state_dict(), model_save_path)

  classifier_module.push_to_hub("fahimfarhan/mqtl-classifier-model")
  pass


if __name__ == "__main__":
  dataset_folder_prefix = "inputdata/"
  pytorch_model = MQtlDnaBERT6Classifier()
  start_bert(classifier_model=pytorch_model, model_save_path=f"weights_{pytorch_model.model_name}.pth",
             criterion=ReshapedBCEWithLogitsLoss(), WINDOW=200, batch_size=4,
             dataset_folder_prefix=dataset_folder_prefix, max_epochs=2)
  pass

import pandas as pd
import torch
from sklearn import metrics
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BatchEncoding
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

# load the DNABert tokenizer, and model
dna_bert_6 = "zhihan1996/DNA_bert_6"  # works on my laptop!

bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=dna_bert_6)
bert_model = BertModel.from_pretrained(pretrained_model_name_or_path=dna_bert_6)

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available()
    else "cpu"
  )


def preprocess_dna(sequence, k=6):
  tokens = [sequence[i:i + k] for i in range(len(sequence) - k + 1)]
  return tokens


def run1():
  dna_sequence = "ATCGTAGCTAGCTAGCTGACT"
  tokens: list[str] = preprocess_dna(dna_sequence)
  print(f"{tokens = }")
  print(f"typeoftokens = {type(tokens) = }")
  encoded_input: BatchEncoding = bert_tokenizer(tokens, return_tensors="pt", is_split_into_words=True)
  print(f"{encoded_input = }")
  print(f"{type(encoded_input) = }")
  """
    tokens = ['ATCGTA', 'TCGTAG', 'CGTAGC', 'GTAGCT', 'TAGCTA', 'AGCTAG', 'GCTAGC', 'CTAGCT', 'TAGCTA', 'AGCTAG',
              'GCTAGC', 'CTAGCT', 'TAGCTG', 'AGCTGA', 'GCTGAC', 'CTGACT']
    typeoftokens = type(tokens) = <

    class 'list'>

    encoded_input = {'input_ids': tensor([[2, 441, 1752, 2899, 3390, 1257, 920, 3667, 2366, 1257, 920, 3667,
                                           2366, 1260, 929, 3703, 2510, 3]]),
                     'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                     'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
    type(encoded_input) = <

    class 'transformers.tokenization_utils_base.BatchEncoding'>
    """
  input_ids = encoded_input["input_ids"]
  attention_mask = encoded_input["attention_mask"]
  print(f"{input_ids = }")
  print(f"{attention_mask = }")
  bert_outputs: BaseModelOutputWithPoolingAndCrossAttentions = bert_model(**encoded_input)
  """
    BaseModelOutputWithPoolingAndCrossAttentions = { 
        last_hidden_state: tensor, 
        pooler_output: tensor, 
        grad_fn, 
        hidden_states,
        past_key_values,
        attentions,
        cross_attentions
    }
    """
  print(f"{bert_outputs = }")
  print(f"{type(bert_outputs) = }")
  # Extract the attention scores (optional)
  # If you need the attention scores for further analysis
  attention_scores = bert_outputs.attentions if 'attentions' in bert_outputs else None
  print(f"{attention_scores = }")
  # Extract the last hidden state
  last_hidden_state = bert_outputs.last_hidden_state

  print("Last Hidden State Shape:", last_hidden_state.shape)


class AttentionLayer(nn.Module):
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
    return context_vector


class DNAClassifierModel(nn.Module):
  def __init__(self, bert_model, hidden_size, num_classes, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.bert_model = bert_model
    self.attention = AttentionLayer(hidden_size)
    self.classifier = nn.Linear(hidden_size, num_classes)
    pass

  def forwardV1Failed(self, encoded_input_x: BatchEncoding):
    input_ids: torch.tensor = encoded_input_x["input_ids"]
    token_type_ids: torch.tensor = encoded_input_x["token_type_ids"]
    attention_mask: torch.tensor = encoded_input_x["attention_mask"]
    """
       bert_output: BaseModelOutputWithPoolingAndCrossAttentions = self.bert_model(
         input_ids=input_ids,
         attention_mask=attention_mask,
         token_type_ids=token_type_ids
       )
       """
    # looks like equivalent to the next line...
    bert_output: BaseModelOutputWithPoolingAndCrossAttentions = self.bert_model(**encoded_input_x)

    last_hidden_state = bert_output.last_hidden_state
    context_vector = self.attention(last_hidden_state)
    y = self.classifier(context_vector)
    return y

  def forwardV2Ok(self, input_ids: torch.tensor, attention_mask: torch.tensor, token_type_ids):
    # input_ids: torch.tensor = encoded_input_x["input_ids"]
    # token_type_ids: torch.tensor = encoded_input_x["token_type_ids"]
    # attention_mask: torch.tensor = encoded_input_x["attention_mask"]
    """
       bert_output: BaseModelOutputWithPoolingAndCrossAttentions = self.bert_model(
         input_ids=input_ids,
         attention_mask=attention_mask,
         token_type_ids=token_type_ids
       )
       """
    # looks like equivalent to the next line...
    inputs_embeds = None
    if input_ids is not None and inputs_embeds is not None:
      raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
      input_shape = input_ids.size()
      print(f"1 { input_shape = }")
    elif inputs_embeds is not None:
      input_shape = inputs_embeds.size()[:-1]
      print(f"2 { input_shape = }")
    else:
      raise ValueError("You have to specify either input_ids or inputs_embeds")

    # torch.Size([128, 1, 512])
    print(f"3 {input_ids = }")
    input_ids = input_ids.squeeze(dim=1).to(DEVICE)
    print(f"4 {input_ids}")

    print(f"5 attention shape: {attention_mask.shape = }")
    print(f"5 attention size: {attention_mask.size = }")

    attention_mask = attention_mask.squeeze(dim=1).to(DEVICE)

    print(f"5 token_type_ids shape: {token_type_ids.shape = }")
    print(f"5 token_type_ids size: {token_type_ids.size = }")

    token_type_ids = token_type_ids.squeeze(dim=1).to(DEVICE)

    bert_output: BaseModelOutputWithPoolingAndCrossAttentions = self.bert_model(
      input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids
    )

    last_hidden_state = bert_output.last_hidden_state
    context_vector = self.attention(last_hidden_state)
    y = self.classifier(context_vector)
    return y

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
    context_vector = self.attention(last_hidden_state)
    y = self.classifier(context_vector)
    return y



class DNABERTDataset(Dataset):
  def __init__(self, df: pd.DataFrame, bert_tokenizer1: BertTokenizer, k: int = 6):
    self.df = df
    self.bert_tokenizer = bert_tokenizer1
    self.k = k
    self.length = len(df["Sequence"])
    pass

  def __len__(self):
    return self.length

  def preprocess(self, sequence: str) -> list[str]:
    tokens = [sequence[i:i + self.k] for i in range(len(sequence) - self.k + 1)]
    return tokens

  def __getitem__(self, idx) -> (BatchEncoding, float):
    sequence = self.df["Sequence"][idx]
    label = float(self.df["class"][idx])
    tokens: list[str] = self.preprocess(sequence)
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


class TrainValidDataset(Dataset):
  def __init__(self, train_ds: Dataset, valid_ds: Dataset):
    self.train_ds = train_ds
    self.valid_ds = valid_ds  # or use it as test dataset
    pass


class MQtlNeuralNetClassifier(NeuralNetClassifier):
  def get_split_datasets(self, X, y=None, **fit_params):
    # overriding this function
    dataset = self.get_dataset(X, y)
    if isinstance(X, TrainValidDataset):
      dataset_train, dataset_valid = X.train_ds, X.valid_ds
      return dataset_train, dataset_valid
    raise AssertionError("X is not a TrainValidDataset!")


# Ensure targets are reshaped properly
class ReshapedBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def forward(self, input, target):
        return super().forward(input.squeeze(), target.float())

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
  df = pd.read_csv("data64random.csv")
  # dataset = DNABERTDataset(df, bert_tokenizer)

  train_data, temp_data = train_test_split(df, test_size=0.2, random_state=42)
  val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

  train_dataset = DNABERTDataset(df, bert_tokenizer)
  val_dataset = DNABERTDataset(df, bert_tokenizer)
  test_dataset = DNABERTDataset(df, bert_tokenizer)

  batch_size = 16

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  # for ith_batch in train_loader:
  #   encoded_input_x, label = ith_batch[0], ith_batch[1]
  #   print(f"{label = }, {encoded_input_x = }")

  m_optimizer = torch.optim.Adam  # (pytorch_model.parameters(), lr=1e-4, weight_decay=1e-5)
  m_loss = ReshapedBCEWithLogitsLoss() # nn.BCEWithLogitsLoss()

  pytorch_model = DNAClassifierModel(bert_model, 768, 1)

  net = MQtlNeuralNetClassifier(
    pytorch_model,
    max_epochs=5,
    criterion=m_loss,
    optimizer=m_optimizer,
    lr=0.01,
    # decay=0.01,
    # momentum=0.9,
    batch_size=4,  # Specify the batch size here
    device=DEVICE,
    classes=["no_mqtl", "yes_mqtl"],
    verbose=True,
    callbacks=get_callbacks()
  )

  train_valid_dataset = TrainValidDataset(train_ds=train_dataset, valid_ds=val_dataset)

  net.fit(train_valid_dataset, y=None)
  pass


if __name__ == '__main__':
  # run1()
  start()
  pass

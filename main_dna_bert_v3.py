import logging

import torch
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModel
from torch import nn
from torch import optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

timber = logging.getLogger()
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)  # change to level=logging.DEBUG to print more logs...


def preprocess_dna(sequence, k=6):
  tokens = [sequence[i:i + k] for i in range(len(sequence) - k + 1)]
  return tokens


def example_code():
  dna_bert_2_117m = "zhihan1996/DNABERT-2-117M"
  dna_bert_6 = "zhihan1996/DNA_bert_6"  # works on my laptop!
  pretrained_model_name = dna_bert_6
  tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name, trust_remote_code=True)
  model = AutoModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_name, trust_remote_code=True)

  dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
  inputs = tokenizer(dna, return_tensors='pt')["input_ids"]
  hidden_states = model(inputs)[0]  # [1, sequence_length, 768]

  # embedding with mean pooling
  embedding_mean = torch.mean(hidden_states[0], dim=0)
  print(embedding_mean.shape)  # expect to be 768

  # embedding with max pooling
  embedding_max = torch.max(hidden_states[0], dim=0)[0]
  print(embedding_max.shape)  # expect to be 768
  pass


class Attention(nn.Module):
  def __init__(self, hidden_size):
    super(Attention, self).__init__()
    self.attention = nn.Linear(hidden_size, 1)

  def forward(self, hidden_states):
    # Apply linear layer
    attn_weights = self.attention(hidden_states)
    # Apply softmax to get attention scores
    attn_weights = torch.softmax(attn_weights, dim=1)
    # Apply attention weights to hidden states
    context_vector = torch.sum(attn_weights * hidden_states, dim=1)
    return context_vector


def example_attention(last_hidden_state):
  # Initialize the attention layer
  attention_layer = Attention(hidden_size=768)  # DNABERT hidden size is typically 768

  # Apply the attention layer to the last hidden state
  context_vector = attention_layer(last_hidden_state)

  print("Context Vector Shape:", context_vector.shape)
  pass


# fine tune the model
class DNABERTClassifier(nn.Module):
  def __init__(self, bert_model, hidden_size, num_classes, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.bert_model = bert_model
    self.attention = Attention(hidden_size)
    self.classifier = nn.Linear(in_features=hidden_size, out_features=num_classes)
    pass

  def forward(self, input_ids, attention_mask):
    bert_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden_state = bert_outputs.last_hidden_state
    context_vector = self.attention(last_hidden_state)
    y = self.classifier(context_vector)
    return y


def start_dna_bert_classifier():
  pass


class MyDataset(Dataset):
  def __init__(self, df, bert_tokenizer, k=6):
    self.df = df
    self.bert_tokenizer = bert_tokenizer
    self.k = k
    pass

  def __len__(self):
    sequences = self.df["sequence"]
    return len(sequences)

  def preprocess_dna(self, sequence):
    tokens = [sequence[i:i + self.k] for i in range(len(sequence) - self.k + 1)]
    return tokens

  def __getitem__(self, idx):
    sequences = self.df["sequence"]
    labels = self.df["label"]

    # print(f"{sequences = }")

    sequence = sequences.iloc[idx]
    print(f"{sequence = }")
    label = labels.iloc[idx]
    tokens = self.preprocess_dna(sequence)
    encoded_input = self.bert_tokenizer(tokens, return_tensors='pt', is_split_into_words=True, padding='max_length',
                                        truncation=True, max_length=512)

    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']
    return input_ids, attention_mask, label


def start():
  dna_bert_6 = "zhihan1996/DNA_bert_6"  # works on my laptop!
  pretrained_model_name = dna_bert_6
  tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name, trust_remote_code=True)
  model = AutoModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_name, trust_remote_code=True)

  dna_sequence = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"

  # Tokenize the DNA sequence
  tokens = preprocess_dna(dna_sequence)
  encoded_input = tokenizer(tokens, return_tensors='pt', is_split_into_words=True)

  # Get the model outputs
  outputs = model(**encoded_input)

  # Extract the attention scores (optional)
  # If you need the attention scores for further analysis
  attention_scores = outputs.attentions if 'attentions' in outputs else None

  # Extract the last hidden state
  last_hidden_state = outputs.last_hidden_state

  print("Last Hidden State Shape:", last_hidden_state.shape)

  # Initialize the classifier
  classifier = DNABERTClassifier(model, hidden_size=768, num_classes=2)

  # Example data
  file_seq_len = 64
  # Data preparation (use your actual dataset)
  data = pd.read_csv(f"data{file_seq_len}random.csv")

  data["sequence"] = data["Sequence"]
  data["label"] = data["class"]

  # print(data["sequence"])
  # print(data["label"])

  train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
  val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

  dna_bert_name = "zhihan1996/DNA_bert_6"

  train_dataset = MyDataset(df=train_data, bert_tokenizer=tokenizer)
  val_dataset = MyDataset(df=val_data, bert_tokenizer=tokenizer)
  test_dataset = MyDataset(df=test_data, bert_tokenizer=tokenizer)

  batch_size = 16

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  criterion = nn.CrossEntropyLoss()
  optimizer = Adam(model.parameters(), lr=1e-5)

  train_model(model, train_loader, val_loader, criterion=criterion, optimizer=optimizer, num_epochs=2)

  pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            keys, values, labels = batch
            # keys = keys.to(device)
            values = values.to(device)

            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(keys, values)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_loss = evaluate_model(model, val_loader, criterion)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')

def evaluate_model(model, data_loader, criterion):
  model.eval()
  total_loss = 0
  with torch.no_grad():
    for batch in data_loader:
      inputs, labels = batch
      inputs = {key: val.to(device) for key, val in inputs.items()}
      labels = labels.to(device)

      outputs = model(**inputs)
      loss = criterion(outputs, labels)
      total_loss += loss.item()

  avg_loss = total_loss / len(data_loader)
  return avg_loss



def train_model_v1(model, train_loader, val_loader, criterion, optimizer, num_epochs=3):
  best_val_loss = float('inf')
  for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
      input_ids, attention_mask, labels = batch
      input_ids = input_ids.to(device)
      attention_mask = attention_mask.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()
      outputs = model(input_ids, attention_mask)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    val_loss = evaluate_model(model, val_loader, criterion)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      # torch.save(model.state_dict(), 'best_model.pt')
  pass


if __name__ == '__main__':
  start()
  pass

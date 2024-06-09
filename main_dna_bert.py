import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split


class DNADataset(Dataset):
  def __init__(self, sequences, labels, tokenizer, k=6):
    self.sequences = sequences
    self.labels = labels
    self.tokenizer = tokenizer
    self.k = k

  def preprocess_dna(self, sequence):
    tokens = [sequence[i:i + self.k] for i in range(len(sequence) - self.k + 1)]
    return tokens

  def __len__(self):
    return len(self.sequences)

  def __getitem__(self, idx):
    sequence = self.sequences[idx]
    label = self.labels[idx]
    tokens = self.preprocess_dna(sequence)
    encoded_input = self.tokenizer(tokens, return_tensors='pt', is_split_into_words=True, padding='max_length',
                                   truncation=True, max_length=512)
    return {key: val.squeeze() for key, val in encoded_input.items()}, torch.tensor(label)


# Example data
file_seq_len = 64
# Data preparation (use your actual dataset)
data = pd.read_csv(f"data{file_seq_len}random.csv")

data["sequence"] = data["Sequence"]
data["label"] = data["class"]

train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

dna_bert_name = "zhihan1996/DNA_bert_6"

tokenizer = BertTokenizer.from_pretrained(dna_bert_name)

train_dataset = DNADataset(train_data['sequence'].tolist(), train_data['label'].tolist(), tokenizer)
val_dataset = DNADataset(val_data['sequence'].tolist(), val_data['label'].tolist(), tokenizer)
test_dataset = DNADataset(test_data['sequence'].tolist(), test_data['label'].tolist(), tokenizer)

batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

import torch.nn as nn
import torch.optim as optim


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


class DNABERTClassifier(nn.Module):
  def __init__(self, bert_model, hidden_size, num_classes):
    super(DNABERTClassifier, self).__init__()
    self.bert = bert_model
    self.attention = Attention(hidden_size)
    self.classifier = nn.Linear(hidden_size, num_classes)

  def forward(self, input_ids, attention_mask):
    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden_state = outputs.last_hidden_state
    context_vector = self.attention(last_hidden_state)
    logits = self.classifier(context_vector)
    return logits


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the classifier

model = DNABERTClassifier(dna_bert_name, hidden_size=768, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3):
  best_val_loss = float('inf')
  for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
      inputs, labels = batch
      inputs = {key: val.to(device) for key, val in inputs.items()}
      labels = labels.to(device)

      optimizer.zero_grad()
      outputs = model(**inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    val_loss = evaluate_model(model, val_loader, criterion)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')

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


# Training the model

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3)

import torch
from torch import nn
from torch import optim
from transformers import BertModel

dna_bert_name = "zhihan1996/DNA_bert_6"

class DNABERTClassifier(nn.Module):
  def __init__(self, bert_model_name: str = dna_bert_name, hidden_size: int = 768, num_classes: int =1, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.bert: BertModel = BertModel.from_pretrained(pretrained_model_name_or_path=bert_model_name)
    self.attention: nn.Linear = nn.Linear(in_features=hidden_size, out_features=1)
    self.classifier: nn.Linear = nn.Linear(in_features=hidden_size, out_features=num_classes)
    pass

  def forward(self, input_ids, attention_mask):
    output = self.bert(input_ids, attention_mask)  # (input_ids, attention_mask)
    last_hidden_state = output.last_hidden_state
    # apply attention
    attention_weight = torch.softmax(self.attention(last_hidden_state), dim=1)
    context_vector = torch.sum(attention_weight * last_hidden_state, dim=1)
    y = self.classifier(context_vector)
    return y


from skorch import NeuralNetClassifier
from skorch.dataset import Dataset
from transformers import BertTokenizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

tokenizer = BertTokenizer.from_pretrained(dna_bert_name)

file_seq_len = 64
# Data preparation (use your actual dataset)
data = pd.read_csv(f"data{file_seq_len}random.csv")

train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)


class DNADataset(Dataset):
  def __init__(self, sequences, labels, tokenizer, k=6, *args, **kwargs):
    # super().__init__(args, kwargs)
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
    return {key: val.squeeze() for key, val in encoded_input.items()}, label


train_dataset = DNADataset(train_data["Sequence"], train_data["class"], tokenizer)
val_dataset = DNADataset(val_data["Sequence"], val_data["class"], tokenizer)
test_dataset = DNADataset(test_data["Sequence"], test_data["class"], tokenizer)


# why do I need another one? :/
class SkorchDNADataset(DNADataset):
  def __getitem__(self, idx):
    sequence, label = super().__getitem__(idx)
    return {key: val.squeeze().numpy() for key, val in sequence.items()}, label


net = NeuralNetClassifier(
    module=DNABERTClassifier,
    module__bert_model_name=dna_bert_name,
    module__hidden_size=768,
    module__num_classes=2,
    criterion=nn.CrossEntropyLoss,
    optimizer=optim.Adam,
    optimizer__lr=1e-5,
    max_epochs=3,
    batch_size=16,
    iterator_train__shuffle=True,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)


# Fit the model
net.fit(X=SkorchDNADataset(train_data["Sequence"].tolist(), train_data["class"].tolist(), tokenizer), y=None)

# Evaluate on validation set
y_val_true = val_data["class"].tolist()
y_val_pred = net.predict(SkorchDNADataset(val_data["Sequence"].tolist(), val_data["class"].tolist(), tokenizer))
val_accuracy = accuracy_score(y_val_true, y_val_pred)
print(f'Validation Accuracy: {val_accuracy:.4f}')

# Evaluate on test set
y_test_true = test_data["class"].tolist()
y_test_pred = net.predict(SkorchDNADataset(test_data["Sequence"].tolist(), test_data["class"].tolist(), tokenizer))
test_accuracy = accuracy_score(y_test_true, y_test_pred)
print(f'Test Accuracy: {test_accuracy:.4f}')

import logging

import torch
from transformers import AutoTokenizer, AutoModel
from torch import nn
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
    bert_outputs = self.bert_model(input_ids = input_ids, attention_mask = attention_mask)
    last_hidden_state = bert_outputs.last_hidden_state
    context_vector = self.attention(last_hidden_state)
    y = self.classifier(context_vector)
    return y


def start_dna_bert_classifier():
  pass

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

  # Example inputs (input_ids and attention_mask)
  input_ids = encoded_input['input_ids']
  attention_mask = encoded_input['attention_mask']

  # Forward pass
  logits = classifier(input_ids, attention_mask)
  print(logits)
  pass

if __name__ == '__main__':
  start()
  pass
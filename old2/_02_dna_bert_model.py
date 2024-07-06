# pip install torch transformers==4.29 einops # the transformer is new...

# load the dna-bert using xformer library

# preprocess / tokenize

# define the model

import torch
# from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel

# load the dna bert tokenizer, and the model
PRETRAINED_MODEL_NAME = "zhihan1996/DNABERT-2-117M"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="zhihan1996/DNABERT-2-117M", trust_remote_code=True)
model = AutoModel.from_pretrained(pretrained_model_name_or_path="zhihan1996/DNABERT-2-117M", trust_remote_code=True)

# def preprocess_dna(sequence, k=6):
#   tokens = [sequence[i:i + k] for i in range(len(sequence) - k + 1)]
#   return tokens


# docs https://huggingface.co/zhihan1996/DNABERT-2-117M
if __name__ == "__main__":
  # device = torch.device("cpu")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
  inputs = tokenizer(dna, return_tensors='pt')["input_ids"]

  inputs = inputs.to(device)
  model = model.to(device)

  hidden_states = model(inputs)[0]  # [1, sequence_length, 768]

  # embedding with mean pooling
  embedding_mean = torch.mean(hidden_states[0], dim=0)
  print(embedding_mean.shape)  # expect to be 768

  # embedding with max pooling
  embedding_max = torch.max(hidden_states[0], dim=0)[0]
  print(embedding_max.shape)  # expect to be 768

import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers.pipelines.base import Dataset

HUGGING_FACE_PRETRAINED_MODEL_NAME = "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf"

DEVICE = torch.device(
  "cuda:0" if torch.cuda.is_available()
  else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=HUGGING_FACE_PRETRAINED_MODEL_NAME,
                                          trust_remote_code=True)

# Step 3: Define the Model
# model = AutoModelForSequenceClassification.from_pretrained(
#   pretrained_model_name_or_path=HUGGING_FACE_PRETRAINED_MODEL_NAME, num_labels=2, trust_remote_code=True)

model = AutoModelForSequenceClassification.from_pretrained(
  pretrained_model_name_or_path=HUGGING_FACE_PRETRAINED_MODEL_NAME, torch_dtype=torch.bfloat16,
  trust_remote_code=True)

model = model.to(DEVICE)


class DNADataset(Dataset):
  def __init__(self, df: pd.DataFrame, mtokenizer: AutoTokenizer):
    self.df = df
    self.tokenizer = mtokenizer
    self.length = len(df["Sequence"])
    pass

  def __len__(self):
    return 50 # self.length

  def __getitem__(self, idx):
    seq = self.df["Sequence"][idx]
    label = self.df["class"][idx]
    sth = self.tokenizer(seq)
    print(f"{idx = }, {label = }, { type(sth) = }")
    input_ids = sth["input_ids"]
    print(f"{type(input_ids) = }, { len(input_ids) = }")
    first_item = input_ids[0]
    print(f"{ first_item = }, { type(first_item) = }")
    return input_ids, label

df = pd.read_csv("data2000random.csv")
ds = DNADataset(df = df, mtokenizer=tokenizer)
dl = DataLoader(ds)

for input_ids, label in dl:
  # print(data)
  # print(f"{ type(input_ids) = }, { label = }")

  input_ids_to_device = [ input_id.to(DEVICE) for input_id in input_ids]
  label = label.to(DEVICE)
  output = model(input_ids, label)
  print(f"{ output = }, { label = }")

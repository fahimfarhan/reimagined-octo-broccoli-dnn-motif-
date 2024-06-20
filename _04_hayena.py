import torch
from torch import nn
from torch import optim
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

HUGGING_FACE_PRETRAINED_MODEL_NAME = "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=HUGGING_FACE_PRETRAINED_MODEL_NAME, trust_remote_code=True)


# Step 3: Define the Model
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=HUGGING_FACE_PRETRAINED_MODEL_NAME, num_labels=2, trust_remote_code=True)

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available()
    else "cpu"
  )

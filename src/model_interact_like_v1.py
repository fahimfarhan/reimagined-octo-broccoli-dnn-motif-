import torch
from torch import nn
import torch.nn.functional as F
from start import start

"""
Based on the interact model (https://www.biorxiv.org/content/10.1101/2024.01.18.576319v1),
 and their github codebase. code link:
https://github.com/LieberInstitute/INTERACT/blob/master/models/modeling_bert.py#L386
https://github.com/LieberInstitute/INTERACT/blob/master/models/modeling_utils.py#L855

Looks like these are the ones we need...
"""


def create_cnn_relu_norm_layer(num_of_nucleotides=4):
  cnn = nn.Conv1d(in_channels=num_of_nucleotides, out_channels=512, kernel_size=10)
  activation = nn.ReLU()
  batch_norm = nn.BatchNorm1d(512)
  return nn.Sequential(cnn, activation, batch_norm)


class InteractModelLikeMQtlClassifier(nn.Module):
  def __init__(self, num_classes=1, window=200, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.file_name = f"weights_InteractModelLikeMQtlClassifier_seqlen_{window}.pth"
    self.num_classes = num_classes

    # layer_list = [create_cnn_relu_norm_layer() for i in range(0, 3)]
    layer_list = [create_cnn_relu_norm_layer(4), create_cnn_relu_norm_layer(512), create_cnn_relu_norm_layer(512)]
    self.conv_list = nn.ModuleList(layer_list)

    self.pooling_layer = nn.MaxPool1d(kernel_size=2)  # Max pooling layer
    self.batch_norm = nn.BatchNorm1d(512)
    self.dropout = nn.Dropout(0.5)  # Dropout after CNN layers

    # 2. Transformer Encoder Module
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=512,  # Dimension of embedding and attention layers
      nhead=8,  # Number of attention heads
      dim_feedforward=512,  # DNN size
      dropout=0.1,  # Dropout value in the encoder
      activation=F.relu
    )
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=8)  # Stack of 8 layers

    # 3. Final DNN Layer
    self.final_dnn_layer = nn.Sequential(
      nn.Linear(512, num_classes),
      nn.Sigmoid()  # Assuming binary classification, replace with appropriate activation for other tasks
    )

  def forward(self, x):
    hf, hb = x[0], x[1]
    h = hf  # for now ignore the backward sequence
    for layer in self.conv_list:
      h = layer(h)
    h = self.pooling_layer(h)
    h = self.batch_norm(h)
    h = self.dropout(h)

    # Transformer Encoder Module forward pass
    # Input needs to be (S, N, E): (sequence_length, batch_size, embedding_size)
    h = h.permute(2, 0, 1)  # (batch_size, channels, seq_len) -> (seq_len, batch_size, channels)

    h = self.transformer_encoder(h)
    # Convert back for DNN
    h = h.mean(dim=0)  # (seq_len, batch_size, channels) -> (batch_size, channels)

    h = self.final_dnn_layer(h)
    return h


if __name__ == "__main__":
  WINDOW = 200
  interact_like_model = InteractModelLikeMQtlClassifier(window=WINDOW)
  start(classifier_model=interact_like_model, model_save_path=interact_like_model.file_name, WINDOW=WINDOW,
        dataset_folder_prefix="inputdata/", is_debug=True)
  pass

"""
is_debug = true

       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      test_accuracy                 0.5
        test_auc                    0.5
      test_f1_score                 0.0
        test_loss           0.6932778358459473
     test_precision                 0.0
       test_recall                  0.0
"""
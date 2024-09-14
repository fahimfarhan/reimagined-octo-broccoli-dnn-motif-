import numpy as np
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


class DNA_CNN(nn.Module):
  def __init__(self):
    super(DNA_CNN, self).__init__()
    self.num_kernel = 400
    self.one_hot_embedding = nn.Embedding(5, 4, padding_idx=0)
    self.one_hot_embedding.weight.data = torch.from_numpy(np.array([[0., 0., 0., 0.],
                                                                    [1., 0., 0., 0.],
                                                                    [0., 1., 0., 0.],
                                                                    [0., 0., 1., 0.],
                                                                    [0., 0., 0., 1.]])).type(torch.FloatTensor)
    self.one_hot_embedding.weight.requires_grad = False
    self.conv1 = nn.Conv1d(4, self.num_kernel, 10, padding=4)
    self.conv2 = nn.Conv1d(self.num_kernel, self.num_kernel, 10, padding=4)
    self.conv3 = nn.Conv1d(self.num_kernel, self.num_kernel, 10, padding=5)
    self.batch = nn.BatchNorm1d(self.num_kernel)
    self.layer_batch1 = nn.LayerNorm((self.num_kernel, 199))  # originally 2000
    self.layer_batch2 = nn.LayerNorm((self.num_kernel, 198))
    self.layer_batch3 = nn.LayerNorm((self.num_kernel, 199))
    self.dropout = nn.Dropout(p=0.5)
    self.pool = nn.MaxPool1d(20, 20)

  def forward(self, forward_sequence):
    # sequence = self.one_hot_embedding(sequence)
    # sequence = torch.transpose(sequence, 1, 2)
    sequence = forward_sequence
    sequence = F.relu(self.conv1(sequence))
    motif = sequence.clone()
    sequence = self.layer_batch1(sequence)
    sequence = F.relu(self.conv2(sequence))
    sequence = self.layer_batch2(sequence)
    sequence = F.relu(self.conv3(sequence))
    sequence = self.layer_batch3(sequence)
    sequence = self.pool(sequence)
    # sequence = sequence.view(-1,self.num_kernel)
    sequence = self.batch(sequence)
    sequence = self.dropout(sequence)
    return sequence, motif


class InteractModelLikeMQtlClassifierV2(nn.Module):
  def __init__(self, num_classes=1, window=200, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.file_name = f"weights_InteractModelLikeMQtlClassifierV2_seqlen_{window}.pth"
    self.num_classes = num_classes

    self.dna_cnn = DNA_CNN()

    # 2. Transformer Encoder Module
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=400,  # Dimension of embedding and attention layers
      nhead=8,  # Number of attention heads
      dim_feedforward=512,  # DNN size
      dropout=0.1,  # Dropout value in the encoder
      activation=F.relu
    )
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=8)  # Stack of 8 layers

    # 3. Final DNN Layer
    self.final_dnn_layer = nn.Sequential(
      nn.Linear(400, num_classes),
      nn.Sigmoid()  # Assuming binary classification, replace with appropriate activation for other tasks
    )

  def forward(self, x):
    hf, hb = x[0], x[1]
    h = hf  # for now ignore the backward sequence
    some_sequence, some_motif = self.dna_cnn(h)
    h = some_sequence
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
  interact_like_model = InteractModelLikeMQtlClassifierV2(window=WINDOW)
  start(classifier_model=interact_like_model, model_save_path=interact_like_model.file_name, WINDOW=WINDOW,
        dataset_folder_prefix="inputdata/", is_debug=True)
  pass

"""
is_debug = True
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      test_accuracy                 0.5
        test_auc                    0.5
      test_f1_score                 0.0
        test_loss           0.6932435631752014
     test_precision                 0.0
       test_recall                  0.0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

"""
from torch import nn
import torch

import torch.nn.functional as F

import mycolors
from extensions import timber, create_conv_sequence
from start import start

WINDOW = 200


class MQTlClassifierFlattenConvLstm(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes, *args, **kwargs):
    # CNN layer to capture local patterns
    super().__init__(*args, **kwargs)
    self.model_name = f"weights_MQTlClassifierFlattenConvLstm_seqlen_{WINDOW}.pth"

    self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=8, stride=1, padding=4)
    self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8, stride=1, padding=4)

    # LSTM to capture long-range dependencies
    self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True)

    # Fully connected layers
    self.fc1 = nn.Linear(hidden_size * 2, 128)
    self.fc2 = nn.Linear(128, num_classes)

  def forward(self, x):
    forward_seq = x[0]
    reverse_seq = x[1]
    # Concatenate forward and reverse sequences
    h = torch.cat([forward_seq, reverse_seq], dim=1)  # Shape [batch_size, 2, sequence_length, 4]

    # Reshape to feed into CNN: [batch_size, 2, sequence_length]
    h = h.view(h.size(0), 2, -1)

    # CNN layers for local patterns
    h = F.relu(self.conv1(h))  # Shape [batch_size, 32, sequence_length]
    h = F.relu(self.conv2(h))  # Shape [batch_size, 64, sequence_length]
    h = h.permute(0, 2, 1)  # Change to [batch_size, sequence_length, 64] for LSTM

    # LSTM for long-range patterns
    lstm_out, _ = self.lstm(h)  # Shape [batch_size, sequence_length, hidden_size * 2]

    # Mean pooling over the sequence length dimension
    lstm_out = lstm_out.mean(dim=1)  # Shape [batch_size, hidden_size * 2]

    # Fully connected layers for classification
    out = F.relu(self.fc1(lstm_out))
    out = self.fc2(out)  # Final output [batch_size, num_classes]

    return out


if __name__ == "__main__":
  # Example usage:
  model = MQTlClassifierFlattenConvLstm(input_size=4, hidden_size=128, num_classes=1)
  start(classifier_model=model, model_save_path=model.model_name, WINDOW=WINDOW, dataset_folder_prefix="inputdata/",
        is_debug=True)
  pass

"""
is_debug = True

     Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      test_accuracy                 0.5
        test_auc                    0.5
      test_f1_score                 0.0
        test_loss           0.6932870149612427
     test_precision                 0.0
       test_recall                  0.0
"""
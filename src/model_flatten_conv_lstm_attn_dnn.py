from torch import nn
import torch

import torch.nn.functional as F

import mycolors
from extensions import timber, create_conv_sequence
from start import start

WINDOW = 200


class MQTlClassifierFlattenConvLstmAttn(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes, *args, **kwargs):
    # CNN layer to capture local patterns
    super().__init__(*args, **kwargs)
    self.model_name = f"weights_MQTlClassifierFlattenConvLstmAttn_seqlen_{WINDOW}.pth"

    self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=8, stride=1, padding=4)
    self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8, stride=1, padding=4)

    # LSTM to capture long-range dependencies
    self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True)

    # Attention layer (optional)
    self.attention = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=4)

    # Fully connected layers
    self.fc1 = nn.Linear(hidden_size * 2, 128)
    self.fc2 = nn.Linear(128, num_classes)

  def forward(self, x):
    hf = x[0]
    hb = x[1]

    h = torch.concatenate(tensors=(hf, hb), dim=2)
    # Assuming forward_seq and reverse_seq are one-hot encoded [batch_size, sequence_length, 4]
    # Concatenate forward and reverse sequence: [batch_size, 2, sequence_length, 4]
    # x = torch.cat([forward_seq, reverse_seq], dim=1)  # Combining forward and reverse

    # Reshape to feed into CNN: merge 4 nucleotides from forward and reverse
    # [batch_size, 2, sequence_length // 4, 16] -> merging 4 nucleotides each from both seqs
    h = h.view(h.size(0), 2, -1)  # Shape [batch_size, 2, sequence_length]

    # CNN layers for local patterns
    h = F.relu(self.conv1(h))  # Shape [batch_size, 32, sequence_length // 4]
    h = F.relu(self.conv2(h))  # Shape [batch_size, 64, sequence_length // 4]
    h = h.permute(0, 2, 1)  # Change to [batch_size, sequence_length // 4, 64] for LSTM

    # LSTM for long-range patterns
    lstm_out, _ = self.lstm(h)  # Shape [batch_size, sequence_length // 4, hidden_size * 2]

    # Attention mechanism (optional step)
    attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
    attn_output = attn_output.mean(dim=1)  # Mean pooling over sequence length

    # Fully connected layers for classification
    out = F.relu(self.fc1(attn_output))
    out = self.fc2(out)  # Final output [batch_size, num_classes]
    return out


if __name__ == "__main__":
  # Example usage:
  model = MQTlClassifierFlattenConvLstmAttn(input_size=4, hidden_size=128, num_classes=1)
  start(classifier_model=model, model_save_path=model.model_name, WINDOW=WINDOW,
        dataset_folder_prefix="inputdata/", is_debug=True)
  pass

"""
is_debug = True

     Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      test_accuracy                 0.5
        test_auc                    0.5
      test_f1_score                 0.0
        test_loss           0.6934347748756409
     test_precision                 0.0
       test_recall                  0.0
"""
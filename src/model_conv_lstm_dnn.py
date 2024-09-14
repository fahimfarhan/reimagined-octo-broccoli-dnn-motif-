import torch
import torch.nn as nn
import torch.nn.functional as F
from start import start


class MQtlClassifierCnnLstmDnn(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes, window, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.model_name = f"weights_MQtlClassifierCnnLstmDnn_seqlen_{window}.pth"
    # CNN layers
    self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=8, padding=4)
    self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8, padding=4)

    # LSTM or Transformer for long-range patterns (optional)
    self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=1, batch_first=True)

    # Fully connected layers
    self.fc1 = nn.Linear(hidden_size, 128)
    self.fc2 = nn.Linear(128, num_classes)

  def forward(self, x):
    forward_seq, reverse_seq = x[0], x[1]
    # Process forward sequence
    fwd = F.relu(self.conv1(forward_seq))
    fwd = F.relu(self.conv2(fwd))
    fwd, _ = self.lstm(fwd.permute(0, 2, 1))  # Shape: [batch_size, seq_length, hidden_size]

    # Process reverse sequence
    rev = F.relu(self.conv1(reverse_seq))
    rev = F.relu(self.conv2(rev))
    rev, _ = self.lstm(rev.permute(0, 2, 1))

    # Combine forward and reverse by max pooling or mean pooling
    combined = torch.max(fwd, rev)  # OR: combined = torch.mean(torch.stack([fwd, rev]), dim=0)

    # Fully connected layers for classification
    out = F.relu(self.fc1(combined[:, -1, :]))  # Take the last output from LSTM
    out = self.fc2(out)

    return out


if __name__ == "__main__":
  WINDOW = 200
  model = MQtlClassifierCnnLstmDnn(input_size=4, hidden_size=128, num_classes=1, window=WINDOW)
  start(classifier_model=model, model_save_path=model.model_name, WINDOW=WINDOW, dataset_folder_prefix="inputdata/",
        is_debug=True)
  pass

"""
debug == True/False

INFO:root:validate_acc = 0.5, validate_auc = 0.5, validate_f1_score = 0.0, validate_precision = 0.0, validate_recall = 0.0
INFO:root:on_train_epoch_end
INFO:root:train_acc = 0.5009163022041321, train_auc = 0.49785539507865906, train_f1_score = 0.3889552354812622, train_precision = 0.4910453259944916, train_recall = 0.32200852036476135

       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      test_accuracy                 0.5
        test_auc                    0.5
      test_f1_score                 0.0
        test_loss           0.6937423348426819
     test_precision                 0.0
       test_recall                  0.0
"""
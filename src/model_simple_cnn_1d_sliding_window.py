import numpy as np
from torch import nn
import torch

import torch.nn.functional as F

import mycolors
from extensions import timber, create_conv_sequence
from start import start


# Example CNN model to process each window
class WindowCNN1DMQTLClassifier(nn.Module):
  def __init__(self, window_size, stride, seq_len, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.file_name = f"weights_WindowCNN1DMQTLClassifier_seqlen_{seq_len}.pth"

    self.window_size = window_size
    self.stride = stride

    # Define CNN layers
    self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=5)
    self.pool = nn.MaxPool1d(2)
    self.fc = nn.Linear(32 * ((window_size - 5 + 1) // 2), 1)  # Adjusted based on window size
    self.output_activation = torch.sigmoid  # not needed if using nn.BCEWithLogitsLoss()

  def forward(self, x):
    forward, backward = x[0], x[1]
    ohe_sequence_tensor = forward
    # Sliding window operation using unfold (PyTorch built-in)
    windows = ohe_sequence_tensor.unfold(dimension=2, size=self.window_size, step=self.stride)

    # Combine batch and window dimensions
    batch_size, channels, num_windows, window_len = windows.size()

    # Reshape to treat windows as independent batches for CNN processing
    windows = windows.permute(0, 2, 1, 3).contiguous()  # shape = (batch_size, num_windows, channels, window_len)
    windows = windows.view(-1, channels, window_len)  # flatten windows for CNN input

    # CNN layers
    x = self.conv1(windows)  # Convolutional layer
    x = nn.ReLU()(x)
    x = self.pool(x)  # Max-pooling layer

    x = x.view(x.size(0), -1)  # Flatten the output from conv layer for the fully connected layer
    x = self.fc(x)  # Fully connected layer

    # Reshape back to (batch_size, num_windows, output_size)
    x = x.view(batch_size, num_windows, -1)

    # Optionally, you could apply a final aggregation step (e.g., max or average pooling) across windows:
    x = torch.mean(x, dim=1)  # Averaging across windows (or use max pooling)
    x = self.output_activation(x)
    return x


if __name__ == '__main__':
  # Define model with window size and stride
  seq_len1 = 200
  window_size1 = 100
  stride1 = 50
  simple_cnn = WindowCNN1DMQTLClassifier(window_size1, stride1, seq_len=seq_len1)

  start(classifier_model=simple_cnn, model_save_path=simple_cnn.file_name, WINDOW=seq_len1,
        dataset_folder_prefix="inputdata/", is_debug=False)
  pass

"""
is_debug=True
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      test_accuracy         0.9110000133514404
        test_auc            0.9781472086906433
      test_f1_score          0.906855046749115
        test_loss           0.24984027445316315
     test_precision         0.9511525630950928
       test_recall          0.8665000200271606
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""

"""
is_debug=False
INFO:root:on_validation_epoch_end
INFO:root:validate_acc = 0.47874999046325684, validate_auc = 0.4684123396873474, validate_f1_score = 0.40984997153282166, validate_precision = 0.47227656841278076, validate_recall = 0.3619999885559082
INFO:root:on_train_epoch_end
INFO:root:train_acc = 0.5430657267570496, train_auc = 0.5592465400695801, train_f1_score = 0.5057390332221985, train_precision = 0.5421579480171204, train_recall = 0.47390496730804443
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      test_accuracy         0.5007500052452087
        test_auc            0.5023378729820251
      test_f1_score         0.4337964355945587
        test_loss           0.6953898668289185
     test_precision         0.5009823441505432
       test_recall          0.3824999928474426
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""

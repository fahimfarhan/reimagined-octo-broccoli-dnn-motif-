import torch
from torch import nn
import torch.nn.functional as F
from start import start

from extensions import create_conv_sequence, timber
import mycolors


class Cnn1dGlobalPoolingClassifier(nn.Module):
  def __init__(self, seq_len,
               # device,
               in_channel_num_of_nucleotides=4,
               kernel_size_k_mer_motif=4,
               num_filters=32,
               lstm_hidden_size=128,
               dnn_size=512,
               conv_seq_list_size=2,
               *args,
               **kwargs):
    super().__init__(*args, **kwargs)
    self.file_name = f"../weights_Cnn1dGlobalPoolingClassifier.pth"

    self.seq_layer_forward = create_conv_sequence(in_channel_num_of_nucleotides, num_filters,
                                                  kernel_size_k_mer_motif)
    self.seq_layer_backward = create_conv_sequence(in_channel_num_of_nucleotides, num_filters,
                                                   kernel_size_k_mer_motif)

    self.conv1 = nn.Conv1d(in_channels=num_filters, out_channels=64, kernel_size=7, padding=3)
    self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, padding=3)
    self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=15, padding=7, dilation=2)  # Dilated Conv

    # Global Pooling Layer
    self.global_pool = nn.AdaptiveMaxPool1d(1)  # Global Average Pooling

    self.fc1 = nn.Linear(256, 512)  # No need to adjust the input size
    self.fc2 = nn.Linear(512, 1)
    self.output_activation = torch.sigmoid  # not needed if using nn.BCEWithLogitsLoss()

    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    xf, xb = x[0], x[1]

    hf = self.seq_layer_forward(xf)
    timber.debug(mycolors.red + f"1{ hf.shape = }")
    hb = self.seq_layer_backward(xb)
    timber.debug(mycolors.green + f"2{ hb.shape = }")

    x = torch.concatenate(tensors=(hf, hb), dim=2)
    timber.debug(mycolors.yellow + f"4{ x.shape = } concat")

    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = self.global_pool(x)  # Apply Global Pooling
    x = x.view(x.size(0), -1)  # Flatten (will be [batch_size, 256])
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    x = self.output_activation(x)
    return x


if __name__ == '__main__':
  WINDOW = 200
  pytorch_model = Cnn1dGlobalPoolingClassifier(WINDOW)
  start(pytorch_model, pytorch_model.file_name, WINDOW=WINDOW, dataset_folder_prefix="inputdata/")
  pass

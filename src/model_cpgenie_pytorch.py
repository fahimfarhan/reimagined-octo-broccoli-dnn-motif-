import torch.nn.functional as F
from wandb.wandb_torch import torch

from start import *

"""
reference: https://github.com/gifford-lab/CpGenie/blob/master/cnn/seq_128x3_5_5_2f_simple.template
"""
class CpGenieMQTLClassifier(nn.Module):
  def __init__(self,
               seq_len,
               in_channel_num_of_nucleotides=4,
               kernel_size_k_mer_motif=4,
               num_filters=4,
               lstm_hidden_size=128,
               dnn_size=128,
               conv_seq_list_size=3,
               dropout_rate=0.5, w_maxnorm=3.0, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # input layer
    self.model_name = f"CpGenieMQTLClassifier"

    self.seq_layer_forward = create_conv_sequence(in_channel_num_of_nucleotides, num_filters,
                                                  kernel_size_k_mer_motif)
    self.seq_layer_backward = create_conv_sequence(in_channel_num_of_nucleotides, num_filters,
                                                   kernel_size_k_mer_motif)
    # CpGenie start
    # First Convolutional Block
    self.conv1 = nn.Conv1d(in_channels=4, out_channels=128, kernel_size=5, padding=2)
    self.pool1 = nn.MaxPool1d(kernel_size=5, stride=3)

    # Second Convolutional Block
    self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
    self.pool2 = nn.MaxPool1d(kernel_size=5, stride=3)

    # Third Convolutional Block
    self.conv3 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5, padding=2)
    self.pool3 = nn.MaxPool1d(kernel_size=5, stride=3)

    # flatten layer
    self.flatten = nn.Flatten()

    # Fully Connected Layers
    self.fc1 = nn.Linear( 1024, 64)  # in_features=512 * (seq_len // 27) # todo: repair it later
    self.fc2 = nn.Linear(64, 64)
    self.fc3 = nn.Linear(64, 2)

    # Dropout
    self.dropout = nn.Dropout(p=dropout_rate)

    # Weight constraints (MaxNorm)
    self.maxnorm = nn.utils.weight_norm
    # CpGenie end
    # output layer
    self.fc4 = nn.Linear(2, 1)
    self.output_activation = torch.sigmoid  # not needed if using nn.BCEWithLogitsLoss()

  def forward(self, x):
    xf, xb = x[0], x[1]

    hf = self.seq_layer_forward(xf)
    timber.debug(mycolors.red + f"1{ hf.shape = }")
    hb = self.seq_layer_backward(xb)
    timber.debug(mycolors.green + f"2{ hb.shape = }")

    h = torch.concatenate(tensors=(hf, hb), dim=2)
    timber.debug(mycolors.yellow + f"3{ h.shape = } concat")
    # CpGenie start
    # Apply convolutional layers
    h = F.relu(self.conv1(h))
    timber.debug(mycolors.yellow + f"4{ h.shape = } conv1")
    h = self.pool1(h)
    timber.debug(mycolors.yellow + f"5{ h.shape = } pool1")
    h = F.relu(self.conv2(h))
    timber.debug(mycolors.yellow + f"6{ h.shape = } conv2")
    h = self.pool2(h)
    timber.debug(mycolors.yellow + f"7{ h.shape = } pool2")
    h = F.relu(self.conv3(h))
    timber.debug(mycolors.yellow + f"8{ h.shape = } conv3")
    h = self.pool3(h)
    timber.debug(mycolors.yellow + f"9{ h.shape = } pool3")
    # Flatten the output
    # h = h.view(h.size(0), -1)
    h = self.flatten(h)
    timber.debug(mycolors.yellow + f"10{ h.shape = } flatten")

    # Apply fully connected layers
    h = F.relu(self.fc1(h))
    timber.debug(mycolors.yellow + f"11{ h.shape = } fc1")
    h = self.dropout(h)

    h = F.relu(self.fc2(h))
    timber.debug(mycolors.yellow + f"12{ h.shape = } fc2")
    h = self.dropout(h)

    h = self.fc3(h)
    timber.debug(mycolors.yellow + f"13{ h.shape = } fc3")

    # Apply softmax
    h = F.softmax(h, dim=1)
    # CpGenie end
    timber.debug(mycolors.yellow + f"14{ h.shape = } softmax")
    # output layer
    h = self.fc4(h)
    h = self.output_activation(h)
    return h


if __name__ == "__main__":
  window = 200
  pytorch_model = CpGenieMQTLClassifier(seq_len=window)
  start(classifier_model=pytorch_model, model_save_path=f"weights_{pytorch_model.model_name}.pth", m_optimizer=torch.optim.RMSprop, WINDOW=window, dataset_folder_prefix="inputdata/")
  pass

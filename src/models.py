from torch import nn
from extensions import timber
import mycolors

class SimpleCNN1DmQtlClassification(nn.Module):
  def __init__(self,
               seq_len,
               in_channel_num_of_nucleotides=4,
               kernel_size_k_mer_motif=4,
               dnn_size=998,
               num_filters=1,
               lstm_hidden_size=128,
               *args,
               **kwargs
               ):
    super().__init__(*args, **kwargs)
    self.conv1d = nn.Conv1d(in_channels=in_channel_num_of_nucleotides, out_channels=num_filters, kernel_size=kernel_size_k_mer_motif, stride=2)
    self.activation = nn.ReLU(inplace=True)
    self.pooling = nn.MaxPool1d(kernel_size=kernel_size_k_mer_motif, stride=2)

    self.flatten = nn.Flatten()

    self.dnn = nn.Linear(in_features=dnn_size, out_features=1)
    self.dnn_act = nn.ReLU(inplace=True)
    self.dropout = nn.Dropout(p=0.0)

    self.out = nn.Linear(in_features=1, out_features=1)
    pass

  def create_conv_sequence(self, in_channel_num_of_nucleotides, num_filters, kernel_size_k_mer_motif):
    conv1d = nn.Conv1d(in_channels=in_channel_num_of_nucleotides, out_channels=num_filters,
                       kernel_size=kernel_size_k_mer_motif, stride=2)
    activation = nn.ReLU(inplace=True)
    pooling = nn.MaxPool1d(kernel_size=kernel_size_k_mer_motif, stride=2)

    nn.Sequential(conv1d, activation, pooling)

  def forward(self, x):
    # xf, xb = x[0], x[1]
    # h = self.conv1d(xf)
    timber.debug(mycolors.magenta + f"1{ x.shape = }")
    h = self.conv1d(x)
    timber.debug(mycolors.magenta + f"2{ h.shape = }")
    h = self.activation(h)
    timber.debug(mycolors.magenta + f"3{ h.shape = }")
    h = self.pooling(h)
    timber.debug(mycolors.magenta + f"4{ h.shape = }")

    h = self.flatten(h)
    timber.debug(mycolors.magenta + f"5{ h.shape = }")
    h = self.dnn(h)
    timber.debug(mycolors.magenta + f"6{ h.shape = }")
    h = self.dnn_act(h)
    timber.debug(mycolors.magenta + f"6{ h.shape = }")
    h = self.dropout(h)
    timber.debug(mycolors.magenta + f"8{ h.shape = }")
    h = self.out(h)
    timber.debug(mycolors.magenta + f"9{ h.shape = }")
    y = h
    return y

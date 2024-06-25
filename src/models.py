import torch
from torch import nn
from extensions import timber
import mycolors


class SimpleCNN1DmQtlClassification(nn.Module):
  def __init__(self,
               seq_len,
               in_channel_num_of_nucleotides=4,
               kernel_size_k_mer_motif=4,
               num_filters=32,
               lstm_hidden_size=128,
               dnn_size = 256,
               *args,
               **kwargs
               ):
    super().__init__(*args, **kwargs)
    # self.conv1d = nn.Conv1d(in_channels=in_channel_num_of_nucleotides, out_channels=num_filters, kernel_size=kernel_size_k_mer_motif, stride=2)
    # self.activation = nn.ReLU(inplace=True)
    # self.pooling = nn.MaxPool1d(kernel_size=kernel_size_k_mer_motif, stride=2)
    self.seq_layer_forward = self.create_conv_sequence(in_channel_num_of_nucleotides, num_filters,
                                                       kernel_size_k_mer_motif)
    self.seq_layer_backward = self.create_conv_sequence(in_channel_num_of_nucleotides, num_filters,
                                                        kernel_size_k_mer_motif)

    tmp = num_filters # * in_channel_num_of_nucleotides
    tmp_num_filters = num_filters
    # size = seq_len * 2
    # self.conv_seq_0 = self.create_conv_sequence(tmp, tmp_num_filters, kernel_size_k_mer_motif)  # output_size0 = size / kernel_size_k_mer_motif
    # self.conv_seq_1 = self.create_conv_sequence(tmp, tmp_num_filters, kernel_size_k_mer_motif)  # output_size1 = output_size0 / kernel_size_k_mer_motif
    # self.conv_seq_2 = self.create_conv_sequence(tmp, tmp_num_filters, kernel_size_k_mer_motif)  # output_size2 = output_size1 / kernel_size_k_mer_motif

    self.flatten = nn.Flatten()

    dnn_in_features = num_filters * int(seq_len / kernel_size_k_mer_motif) * 2
    # two because forward_sequence,and backward_sequence

    # dnn_in_features = num_filters * int(seq_len / (kernel_size_k_mer_motif ** 4)) * 2
    # dnn_in_features = num_filters * int(seq_len / (kernel_size_k_mer_motif ** 2)) * 2
    self.dnn = nn.Linear(in_features=dnn_in_features, out_features=dnn_size)
    self.dnn_act = nn.ReLU(inplace=True)
    self.dropout = nn.Dropout(p=0.33)

    self.out = nn.Linear(in_features=dnn_size, out_features=1)
    self.sigmoid = torch.sigmoid
    pass

  def create_conv_sequence(self, in_channel_num_of_nucleotides, num_filters, kernel_size_k_mer_motif):
    conv1d = nn.Conv1d(in_channels=in_channel_num_of_nucleotides, out_channels=num_filters,
                       kernel_size=kernel_size_k_mer_motif, padding="same") # stride = 2, just dont use stride, keep it simple for now
    activation = nn.ReLU(inplace=True)
    pooling = nn.MaxPool1d(kernel_size=kernel_size_k_mer_motif)  # stride = 2, just dont use stride, keep it simple for now

    return nn.Sequential(conv1d, activation, pooling)

  def forward(self, x):
    xf, xb = x[0], x[1]

    hf = self.seq_layer_forward(xf)
    timber.debug(mycolors.magenta + f"1{ hf.shape = }")
    hb = self.seq_layer_backward(xb)
    timber.debug(mycolors.magenta + f"2{ hb.shape = }")

    h = torch.concatenate(tensors=(hf, hb), dim=2)
    timber.debug(mycolors.magenta + f"4{ h.shape = } concat")

    # todo: use more / less layers and see what happens
    # h = self.conv_seq_0(h)
    # timber.debug(mycolors.magenta + f"4{ h.shape = } conv_seq_0")

    # h = self.conv_seq_1(h)
    # timber.debug(mycolors.magenta + f"4{ h.shape = } conv_seq_1")

    # h = self.conv_seq_2(h)
    # timber.debug(mycolors.magenta + f"4{ h.shape = } conv_seq_2")

    # h = self.conv1d(xf)
    # timber.debug(mycolors.magenta + f"1{ xf.shape = }")
    # h = self.conv1d(xf)
    # timber.debug(mycolors.magenta + f"2{ h.shape = }")
    # h = self.activation(h)
    # timber.debug(mycolors.magenta + f"3{ h.shape = }")
    # h = self.pooling(h)
    # timber.debug(mycolors.magenta + f"4{ h.shape = }")
    h = self.flatten(h)
    timber.debug(mycolors.magenta + f"5{ h.shape = }")
    h = self.dnn(h)
    timber.debug(mycolors.magenta + f"6{ h.shape = }")
    h = self.dnn_act(h)
    timber.debug(mycolors.magenta + f"7{ h.shape = }")
    h = self.dropout(h)
    timber.debug(mycolors.magenta + f"8{ h.shape = }")
    h = self.out(h)
    timber.debug(mycolors.magenta + f"9{ h.shape = }")
    # h = self.sigmoid(h)
    # timber.debug(mycolors.magenta + f"10{ h.shape = }")
    # a sigmoid is already added in the BCEWithLogitsLoss Function. Hence don't use another sigmoid!
    y = h
    return y

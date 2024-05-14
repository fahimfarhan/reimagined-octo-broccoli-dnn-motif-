import logging

import torch.nn as nn

import constants

timber = logging.getLogger()
logging.basicConfig(level=logging.INFO)  # change to level=logging.DEBUG to print more logs...

#  the original
class CNN1D(nn.Module):
  def __init__(self,
               in_channel_num_of_nucleotides=4,
               kernel_size_k_mer_motif=4,
               dnn_size=1024,
               num_filters=1,
               lstm_hidden_size=128,
               *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.conv1d = nn.Conv1d(in_channels=in_channel_num_of_nucleotides, out_channels=num_filters,
                            kernel_size=kernel_size_k_mer_motif, stride=2)
    self.activation = nn.ReLU()
    self.pooling = nn.MaxPool1d(kernel_size=kernel_size_k_mer_motif, stride=2)

    self.flatten = nn.Flatten()
    # linear layer

    self.dnn2 = nn.Linear(in_features=14 * num_filters, out_features=dnn_size)
    self.act2 = nn.Sigmoid()
    self.dropout2 = nn.Dropout(p=0.2)

    self.out = nn.Linear(in_features=dnn_size, out_features=1)
    self.out_act = nn.Sigmoid()

    pass

  def forward(self, x):
    timber.debug(constants.magenta + f"h0: {x}")
    h = self.conv1d(x)
    timber.debug(constants.green + f"h1: {h}")
    h = self.activation(h)
    timber.debug(constants.magenta + f"h2: {h}")
    h = self.pooling(h)
    timber.debug(constants.blue + f"h3: {h}")
    timber.debug(constants.cyan + f"h4: {h}")

    h = self.flatten(h)
    timber.debug(constants.magenta + f"h5: {h},\n shape {h.shape}, size {h.size}")
    h = self.dnn2(h)
    timber.debug(constants.green + f"h6: {h}")

    h = self.act2(h)
    timber.debug(constants.blue + f"h7: {h}")

    h = self.dropout2(h)
    timber.debug(constants.cyan + f"h8: {h}")

    h = self.out(h)
    timber.debug(constants.magenta + f"h9: {h}")

    h = self.out_act(h)
    timber.debug(constants.green + f"h10: {h}")
    # h = (h > 0.5).float()  # <---- should this go here?
    # timber.debug(constants.green + f"h11: {h}")

    return h



def get_stackoverflow_model():
  n_features = 4

  stack_overflow_model = nn.Sequential(
    # > (batch, seq_len, channels)

    # Initial wide receptive field (and it matches length of the pattern)
    nn.Conv1d(in_channels=n_features, out_channels=4, kernel_size=10, padding='same'),
    nn.ReLU(),
    nn.BatchNorm1d(num_features=4),

    # Conv block 1 doubles features
    nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3),
    nn.ReLU(),
    nn.BatchNorm1d(num_features=8),

    # Conv block 2, then maxpool
    nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3),
    nn.ReLU(),
    nn.BatchNorm1d(num_features=8),

    # Output layer: flatten, linear
    nn.MaxPool1d(kernel_size=2, stride=2),  # batch, feat, seq
    nn.Flatten(start_dim=1),  # batch, feat*seq
    nn.Linear(8 * 30, 1),
  )
  return stack_overflow_model


#  v2
class CNN1Dv2(nn.Module):
  def __init__(self, in_channel_num_of_nucleotides=4, kernel_size_k_mer_motif=4, dnn_size=512, num_filters=1,
               lstm_hidden_size=128, *args, **kwargs):
    super().__init__(*args, **kwargs)
    pass

    # input / low level features extracting conv layer
    self.conv1d0 = nn.Conv1d(in_channels=in_channel_num_of_nucleotides, out_channels=in_channel_num_of_nucleotides, kernel_size=4)
    self.activation0 = nn.ReLU()
    self.batch_norm0 = nn.BatchNorm1d(num_features=in_channel_num_of_nucleotides)

    # mid level features extracting conv layer
    double_features = 2 * in_channel_num_of_nucleotides

    self.conv1d1 = nn.Conv1d(in_channels=in_channel_num_of_nucleotides, out_channels=double_features,
                             kernel_size=4, padding="same")
    self.activation1 = nn.ReLU()
    self.batch_norm1 = nn.BatchNorm1d(num_features=double_features)

    # high level features extracting conv layer
    self.conv1d2 = nn.Conv1d(in_channels=double_features, out_channels=double_features,
                             kernel_size=4, padding="same")
    self.activation2 = nn.ReLU()
    self.batch_norm2 = nn.BatchNorm1d(num_features=double_features)

    # output layers
    self.pooling = nn.MaxPool1d(kernel_size=2, stride=2) # batch, feat, seq
    self.flatten = nn.Flatten() # batch, feat*seq
    self.dnn = nn.Linear(240, dnn_size)
    # the stackoverflow answer stopped here. I  am adding the following layers. Maybe they'll come in handy with lstm

    self.dnn_act = nn.ReLU()

    self.dropout = nn.Dropout(p=0.2)

    self.out = nn.Linear(in_features=dnn_size, out_features=1)
    self.out_act = nn.Sigmoid()   # eta na dile valo perform kore! :?
    pass

  def forward(self, x):
    h = self.conv1d0(x)
    timber.debug(f"0 h.shape: {h.shape}")
    h = self.activation0(h)
    timber.debug(f"1 h.shape: {h.shape}")
    h = self.batch_norm0(h)
    timber.debug(f"2 h.shape: {h.shape}")

    h = self.conv1d1(h)
    timber.debug(f"3 h.shape: {h.shape}")
    h = self.activation1(h)
    timber.debug(f"4 h.shape: {h.shape}")
    h = self.batch_norm1(h)
    timber.debug(f"5 h.shape: {h.shape}")

    h = self.conv1d2(h)
    timber.debug(f"6 h.shape: {h.shape}")
    h = self.activation2(h)
    timber.debug(f"7 h.shape: {h.shape}")
    h = self.batch_norm2(h)
    timber.debug(f"8 h.shape: {h.shape}")

    h = self.pooling(h)
    timber.debug(f"9 h.shape: {h.shape}")
    h = self.flatten(h)
    timber.debug(f"10 h.shape: {h.shape}")
    h = self.dnn(h)
    timber.debug(f"11 h.shape: {h.shape}")
    h = self.dnn_act(h)
    timber.debug(f"12 h.shape: {h.shape}")
    h = self.dropout(h)
    timber.debug(f"13 h.shape: {h.shape}")

    h = self.out(h)
    timber.debug(f"14 h.shape: {h.shape}")
    h = self.out_act(h)
    timber.debug(f"15 h.shape: {h.shape}")
    y = h
    return y


# v3
class CnnLstm1D(nn.Module):
  def __init__(self, in_channel_num_of_nucleotides=4, kernel_size_k_mer_motif=4, dnn_size=512, num_filters=1,
               lstm_hidden_size=128, *args, **kwargs):
    super().__init__(*args, **kwargs)
    pass

    # input / low level features extracting conv layer
    self.conv1d0 = nn.Conv1d(in_channels=in_channel_num_of_nucleotides, out_channels=in_channel_num_of_nucleotides, kernel_size=4)
    self.activation0 = nn.ReLU()
    self.batch_norm0 = nn.BatchNorm1d(num_features=in_channel_num_of_nucleotides)

    # mid level features extracting conv layer
    double_features = 2 * in_channel_num_of_nucleotides

    self.conv1d1 = nn.Conv1d(in_channels=in_channel_num_of_nucleotides, out_channels=double_features,
                             kernel_size=4, padding="same")
    self.activation1 = nn.ReLU()
    self.batch_norm1 = nn.BatchNorm1d(num_features=double_features)

    # high level features extracting conv layer
    self.conv1d2 = nn.Conv1d(in_channels=double_features, out_channels=double_features,
                             kernel_size=4, padding="same")
    self.activation2 = nn.ReLU()
    self.batch_norm2 = nn.BatchNorm1d(num_features=double_features)

    # output layers
    self.pooling = nn.MaxPool1d(kernel_size=2, stride=2) # batch, feat, seq
    self.flatten = nn.Flatten() # batch, feat*seq

    # lstm
    self.bidirectional_lstm = nn.LSTM(input_size=240, hidden_size=lstm_hidden_size, bidirectional=True)

    self.dnn = nn.Linear(256, dnn_size)
    # the stackoverflow answer stopped here. I  am adding the following layers. Maybe they'll come in handy with lstm

    self.dnn_act = nn.ReLU()

    self.dropout = nn.Dropout(p=0.2)

    self.out = nn.Linear(in_features=dnn_size, out_features=1)
    self.out_act = nn.Sigmoid()   # eta na dile valo perform kore! :?
    pass

  def forward(self, x):
    h = self.conv1d0(x)
    timber.debug(f"0 h.shape: {h.shape}")
    h = self.activation0(h)
    timber.debug(f"1 h.shape: {h.shape}")
    h = self.batch_norm0(h)
    timber.debug(f"2 h.shape: {h.shape}")

    h = self.conv1d1(h)
    timber.debug(f"3 h.shape: {h.shape}")
    h = self.activation1(h)
    timber.debug(f"4 h.shape: {h.shape}")
    h = self.batch_norm1(h)
    timber.debug(f"5 h.shape: {h.shape}")

    h = self.conv1d2(h)
    timber.debug(f"6 h.shape: {h.shape}")
    h = self.activation2(h)
    timber.debug(f"7 h.shape: {h.shape}")
    h = self.batch_norm2(h)
    timber.debug(f"8 h.shape: {h.shape}")

    h = self.pooling(h)
    timber.debug(f"9 h.shape: {h.shape}")
    h = self.flatten(h)
    timber.debug(f"10 h.shape: {h.shape}")

    h, dont_care = self.bidirectional_lstm(h)  # cz the output is a tuple
    timber.debug(f"11 h.shape: {h}")

    h = self.dnn(h)
    timber.debug(f"12 h.shape: {h.shape}")
    h = self.dnn_act(h)
    timber.debug(f"13 h.shape: {h.shape}")
    h = self.dropout(h)
    timber.debug(f"14 h.shape: {h.shape}")

    h = self.out(h)
    timber.debug(f"15 h.shape: {h.shape}")
    # h = self.out_act(h)  # just don't use any activation layer just because every tutorial uses it!
    # timber.debug(f"16 h.shape: {h.shape}")
    y = h
    return y


import logging

import torch.nn as nn
import torch
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


class CNN1DNoOutputActivation(nn.Module):
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
    self.act2 = nn.ReLU()
    self.dropout2 = nn.Dropout(p=0.0)

    self.out = nn.Linear(in_features=dnn_size, out_features=1)
    # self.out_act = nn.ReLU()
    # softmax 0.5
    # sigmoid 0.5
    # relu 0.5
    # no output activation tr_acc = 71.5 %, tr_auc = 69%
    #                     val_acc = 63%,   val_auc = 63%

    # mid_activation ==> ReLU acc = 89%, auc = 64%
    # no dropout tr_auc = 96%, auc = 72%
    # dropout 0 ==> best auc = 77%, acc = 98%
    #         .2  auc 65%, acc = 95%, .5 => auc = 80%, acc = 80%,
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

    # h = self.out_act(h)
    # timber.debug(constants.green + f"h10: {h}")
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
               lstm_hidden_size=128, seq_len=64, *args, **kwargs):
    super().__init__(*args, **kwargs)
    pass

    # input / low level features extracting conv layer
    self.conv1d0 = nn.Conv1d(in_channels=in_channel_num_of_nucleotides, out_channels=in_channel_num_of_nucleotides,
                             kernel_size=kernel_size_k_mer_motif, padding="same")
    self.activation0 = nn.ReLU()
    self.batch_norm0 = nn.BatchNorm1d(num_features=in_channel_num_of_nucleotides)

    # mid level features extracting conv layer
    double_features = 2 * in_channel_num_of_nucleotides

    self.conv1d1 = nn.Conv1d(in_channels=in_channel_num_of_nucleotides, out_channels=double_features,
                             kernel_size=kernel_size_k_mer_motif, padding="same")
    self.activation1 = nn.ReLU()
    self.batch_norm1 = nn.BatchNorm1d(num_features=double_features)

    # high level features extracting conv layer
    self.conv1d2 = nn.Conv1d(in_channels=double_features, out_channels=double_features,
                             kernel_size=kernel_size_k_mer_motif, padding="same")
    self.activation2 = nn.ReLU()
    self.batch_norm2 = nn.BatchNorm1d(num_features=double_features)

    # output layers
    pooling_kernel_stride = 2
    # pooking input seq_len = 64
    self.pooling = nn.MaxPool1d(kernel_size=pooling_kernel_stride,
                                stride=pooling_kernel_stride)  # batch, feat, seq
    # pooling output seq_len = 64/ pks = 32
    self.flatten = nn.Flatten()  # batch, feat*seq

    new_seq_len = int(seq_len / pooling_kernel_stride)
    self.dnn = nn.Linear(double_features * new_seq_len, dnn_size)  # 8 * 32
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
    # h = self.out_act(h)
    # timber.debug(f"15 h.shape: {h.shape}")
    y = h
    return y


# v3
class CnnLstm1D(nn.Module):
  def __init__(self, in_channel_num_of_nucleotides=4, kernel_size_k_mer_motif=8, dnn_size=1024, num_filters=1,
               lstm_hidden_size=128, seq_len=64, *args, **kwargs):
    super().__init__(*args, **kwargs)
    pass

    # input / low level features extracting conv layer
    self.conv1d0 = nn.Conv1d(in_channels=in_channel_num_of_nucleotides, out_channels=in_channel_num_of_nucleotides,
                             kernel_size=kernel_size_k_mer_motif, padding="same")
    # padding = "same" to keep seq_len const for convenience
    self.activation0 = nn.ReLU()
    self.batch_norm0 = nn.BatchNorm1d(num_features=in_channel_num_of_nucleotides)

    # mid level features extracting conv layer
    double_features = 2 * in_channel_num_of_nucleotides

    self.conv1d1 = nn.Conv1d(in_channels=in_channel_num_of_nucleotides, out_channels=double_features,
                             kernel_size=kernel_size_k_mer_motif, padding="same")
    self.activation1 = nn.ReLU()
    self.batch_norm1 = nn.BatchNorm1d(num_features=double_features)

    # high level features extracting conv layer
    self.conv1d2 = nn.Conv1d(in_channels=double_features, out_channels=double_features,
                             kernel_size=kernel_size_k_mer_motif, padding="same")
    self.activation2 = nn.ReLU()
    self.batch_norm2 = nn.BatchNorm1d(num_features=double_features)

    # output layers
    # input_seq_len = 64
    pooling_kernel_stride = 2
    self.pooling = nn.MaxPool1d(kernel_size=pooling_kernel_stride, stride=pooling_kernel_stride) # batch, feat, seq
    # output seq_len = 64 / 2 = 32
    self.flatten = nn.Flatten()  # batch, feat*seq

    # lstm
    self.bidirectional_lstm = nn.LSTM(input_size=double_features * int(seq_len / pooling_kernel_stride),
                                      hidden_size=lstm_hidden_size, bidirectional=True)
    lstm_output_shape = lstm_hidden_size * 2  # size1 = double_features * int(seq_len / pooling_kernel_stride)
    self.dnn = nn.Linear(lstm_output_shape, dnn_size)
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
    timber.debug(f"11 h.shape: {h.shape}")

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


class LambdaLayer(nn.Module):
  """
    tensorflow to pytorch lambda layer: https://discuss.pytorch.org/t/how-to-implement-keras-layers-core-lambda-in-pytorch/5903/2?u=fahimfarhan
    """

  def __init__(self, m_lambda):
    super(LambdaLayer, self).__init__()
    self.m_lambda = m_lambda

  def forward(self, x):
    return self.m_lambda(x)


class ReduceSumLambdaLayer(LambdaLayer):
  def __init__(self, m_lambda=torch.sum, m_dim=2):
    super().__init__(m_lambda)
    self.dim = m_dim  # in tensorflow, this dim was 1 :/

  def forward(self, x):
    return self.m_lambda(input=x, dim=self.dim)  # torch.sum(x,dim= 1)


class TimeDistributed(nn.Module):
  def __init__(self, module):
    super(TimeDistributed, self).__init__()
    self.module = module

  def forward(self, x):
    if len(x.size()) <= 2:
      return self.module(x)
    t, n = x.size(0), x.size(1)
    # merge batch and seq dimensions
    x_reshape = x.contiguous().view(t * n, x.size(2))
    y = self.module(x_reshape)
    # We have to reshape Y
    y = y.contiguous().view(t, n, y.size()[1])
    return y


# v3 failed cz output size expected: [128, 1] actual [128, 8, 1], so we need to remove that 8,
class CnnTDFLstm1D_failed1(nn.Module):
  def __init__(self, in_channel_num_of_nucleotides=4, kernel_size_k_mer_motif=8, dnn_size=1024, num_filters=1,
               lstm_hidden_size=128, seq_len=64, *args, **kwargs):
    super().__init__(*args, **kwargs)
    pass

    # input / low level features extracting conv layer
    self.conv1d0 = nn.Conv1d(in_channels=in_channel_num_of_nucleotides, out_channels=in_channel_num_of_nucleotides,
                             kernel_size=kernel_size_k_mer_motif, padding="same")
    # padding = "same" to keep seq_len const for convenience
    self.activation0 = nn.ReLU()
    self.batch_norm0 = nn.BatchNorm1d(num_features=in_channel_num_of_nucleotides)

    # mid level features extracting conv layer
    double_features = 2 * in_channel_num_of_nucleotides

    self.conv1d1 = nn.Conv1d(in_channels=in_channel_num_of_nucleotides, out_channels=double_features,
                             kernel_size=kernel_size_k_mer_motif, padding="same")
    self.activation1 = nn.ReLU()
    self.batch_norm1 = nn.BatchNorm1d(num_features=double_features)

    # high level features extracting conv layer
    self.conv1d2 = nn.Conv1d(in_channels=double_features, out_channels=double_features,
                             kernel_size=kernel_size_k_mer_motif, padding="same")
    self.activation2 = nn.ReLU()
    self.batch_norm2 = nn.BatchNorm1d(num_features=double_features)

    # output layers
    # input_seq_len = 64
    pooling_kernel_stride = 2
    self.pooling = nn.MaxPool1d(kernel_size=pooling_kernel_stride, stride=pooling_kernel_stride) # batch, feat, seq
    # output seq_len = 64 / 2 = 32
    self.flatten = TimeDistributed(nn.Flatten())  # batch, feat*seq

    # lstm
    lstm_input_size = int(double_features * seq_len / (pooling_kernel_stride * kernel_size_k_mer_motif) )
    self.bidirectional_lstm = nn.LSTM(input_size=lstm_input_size ,
                                      hidden_size=lstm_hidden_size, bidirectional=True)
    lstm_output_shape = lstm_hidden_size * 2  # size1 = double_features * int(seq_len / pooling_kernel_stride)
    self.dnn = nn.Linear(lstm_output_shape, dnn_size)
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
    timber.debug(constants.green + f"9 h.shape: {h.shape}, h.size = {h.size}")
    h = self.flatten(h)
    timber.debug(constants.blue + f"10 h.shape: {h.shape}, h.size = {h.size}")

    h, dont_care = self.bidirectional_lstm(h)  # cz the output is a tuple
    timber.debug(f"11 h.shape: {h.shape}")

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

# v4 add reduced sum. trying 1st possibility [128, 8, 64] -> reducedSum -> [128, 8],  . Ran ok...
class CnnTDFLstm1Dv2(nn.Module):
  def __init__(self, in_channel_num_of_nucleotides=4, kernel_size_k_mer_motif=8, dnn_size=1024, num_filters=1,
               lstm_hidden_size=128, seq_len=64, *args, **kwargs):
    super().__init__(*args, **kwargs)
    pass

    # input / low level features extracting conv layer
    self.conv1d0 = nn.Conv1d(in_channels=in_channel_num_of_nucleotides, out_channels=in_channel_num_of_nucleotides,
                             kernel_size=kernel_size_k_mer_motif, padding="same")
    # padding = "same" to keep seq_len const for convenience
    self.activation0 = nn.ReLU()
    self.batch_norm0 = nn.BatchNorm1d(num_features=in_channel_num_of_nucleotides)

    # mid level features extracting conv layer
    double_features = 2 * in_channel_num_of_nucleotides

    self.conv1d1 = nn.Conv1d(in_channels=in_channel_num_of_nucleotides, out_channels=double_features,
                             kernel_size=kernel_size_k_mer_motif, padding="same")
    self.activation1 = nn.ReLU()
    self.batch_norm1 = nn.BatchNorm1d(num_features=double_features)

    # high level features extracting conv layer
    self.conv1d2 = nn.Conv1d(in_channels=double_features, out_channels=double_features,
                             kernel_size=kernel_size_k_mer_motif, padding="same")
    self.activation2 = nn.ReLU()
    self.batch_norm2 = nn.BatchNorm1d(num_features=double_features)

    # output layers
    # input_seq_len = 64
    pooling_kernel_stride = 2
    self.pooling = ReduceSumLambdaLayer() # nn.MaxPool1d(kernel_size=pooling_kernel_stride, stride=pooling_kernel_stride) # batch, feat, seq
    # output seq_len = 64 / 2 = 32
    self.flatten = TimeDistributed(nn.Flatten())  # batch, feat*seq

    # lstm
    lstm_input_size = kernel_size_k_mer_motif # int(double_features * seq_len / (pooling_kernel_stride * kernel_size_k_mer_motif) )
    self.bidirectional_lstm = nn.LSTM(input_size=lstm_input_size ,
                                      hidden_size=lstm_hidden_size, bidirectional=True)
    lstm_output_shape = lstm_hidden_size * 2  # size1 = double_features * int(seq_len / pooling_kernel_stride)
    self.dnn = nn.Linear(lstm_output_shape, dnn_size)
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
    timber.debug(constants.green + f"9 h.shape: {h.shape}, h.size = {h.size}")
    h = self.flatten(h)
    timber.debug(constants.blue + f"10 h.shape: {h.shape}, h.size = {h.size}")

    h, dont_care = self.bidirectional_lstm(h)  # cz the output is a tuple
    timber.debug(f"11 h.shape: {h.shape}")

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

# v5 add reduced sum. trying 2nd possibility 2. [128, 8, 64] -> reducedSum -> [128, 64] . Ran ok...
class CnnTDFLstm1Dv3(nn.Module):
  def __init__(self, in_channel_num_of_nucleotides=4, kernel_size_k_mer_motif=8, dnn_size=1024, num_filters=1,
               lstm_hidden_size=128, seq_len=64, *args, **kwargs):
    super().__init__(*args, **kwargs)
    pass

    # input / low level features extracting conv layer
    self.conv1d0 = nn.Conv1d(in_channels=in_channel_num_of_nucleotides, out_channels=in_channel_num_of_nucleotides,
                             kernel_size=kernel_size_k_mer_motif, padding="same")
    # padding = "same" to keep seq_len const for convenience
    self.activation0 = nn.ReLU()
    self.batch_norm0 = nn.BatchNorm1d(num_features=in_channel_num_of_nucleotides)

    # mid level features extracting conv layer
    double_features = 2 * in_channel_num_of_nucleotides

    self.conv1d1 = nn.Conv1d(in_channels=in_channel_num_of_nucleotides, out_channels=double_features,
                             kernel_size=kernel_size_k_mer_motif, padding="same")
    self.activation1 = nn.ReLU()
    self.batch_norm1 = nn.BatchNorm1d(num_features=double_features)

    # high level features extracting conv layer
    self.conv1d2 = nn.Conv1d(in_channels=double_features, out_channels=double_features,
                             kernel_size=kernel_size_k_mer_motif, padding="same")
    self.activation2 = nn.ReLU()
    self.batch_norm2 = nn.BatchNorm1d(num_features=double_features)

    # output layers
    # input_seq_len = 64
    pooling_kernel_stride = 1 # since we haven't used MaxPooling, pks is not used. so setting it to 1
    self.pooling = ReduceSumLambdaLayer(m_dim=1)  # nn.MaxPool1d(kernel_size=pooling_kernel_stride, stride=pooling_kernel_stride) # batch, feat, seq
    # output seq_len = 64 / 2 = 32
    self.flatten = TimeDistributed(nn.Flatten())  # batch, feat*seq

    # lstm
    lstm_input_size = int(double_features * seq_len / (pooling_kernel_stride * kernel_size_k_mer_motif) )
    self.bidirectional_lstm = nn.LSTM(input_size=lstm_input_size ,
                                      hidden_size=lstm_hidden_size, bidirectional=True)
    lstm_output_shape = lstm_hidden_size * 2  # size1 = double_features * int(seq_len / pooling_kernel_stride)
    self.dnn = nn.Linear(lstm_output_shape, dnn_size)
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
    timber.debug(constants.green + f"9 h.shape: {h.shape}, h.size = {h.size}")
    h = self.flatten(h)
    timber.debug(constants.blue + f"10 h.shape: {h.shape}, h.size = {h.size}")

    h, dont_care = self.bidirectional_lstm(h)  # cz the output is a tuple
    timber.debug(f"11 h.shape: {h.shape}")

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



"""
time distributed flatten
DEBUG:root:9 h.shape: torch.Size([128, 8, 32]), h.size = <built-in method size of Tensor object at 0x75735c212e10>
DEBUG:root:10 h.shape: torch.Size([128, 8, 32]), h.size = <built-in method size of Tensor object at 0x75735c212ed0>

normal flatten
DEBUG:root:9 h.shape: torch.Size([128, 8, 32]), h.size = <built-in method size of Tensor object at 0x7ed78431fa70>
DEBUG:root:10 h.shape: torch.Size([128, 256]), h.size = <built-in method size of Tensor object at 0x7ed78431f290>

fixed

problem 2:

time distributed flatten: 

DEBUG:root:0 h.shape: torch.Size([128, 4, 64])
DEBUG:root:1 h.shape: torch.Size([128, 4, 64])
DEBUG:root:2 h.shape: torch.Size([128, 4, 64])
DEBUG:root:3 h.shape: torch.Size([128, 8, 64])
DEBUG:root:4 h.shape: torch.Size([128, 8, 64])
DEBUG:root:5 h.shape: torch.Size([128, 8, 64])
DEBUG:root:6 h.shape: torch.Size([128, 8, 64])
DEBUG:root:7 h.shape: torch.Size([128, 8, 64])
DEBUG:root:8 h.shape: torch.Size([128, 8, 64])
DEBUG:root:9 h.shape: torch.Size([128, 8, 32]), h.size = <built-in method size of Tensor object at 0x76db2cf1edb0>
DEBUG:root:10 h.shape: torch.Size([128, 8, 32]), h.size = <built-in method size of Tensor object at 0x76db2cf1ee70>
DEBUG:root:11 h.shape: torch.Size([128, 8, 256])
DEBUG:root:12 h.shape: torch.Size([128, 8, 1024])
DEBUG:root:13 h.shape: torch.Size([128, 8, 1024])
DEBUG:root:14 h.shape: torch.Size([128, 8, 1024])
DEBUG:root:15 h.shape: torch.Size([128, 8, 1])

normal flatten
DEBUG:root:0 h.shape: torch.Size([128, 4, 64])
DEBUG:root:1 h.shape: torch.Size([128, 4, 64])
DEBUG:root:2 h.shape: torch.Size([128, 4, 64])
DEBUG:root:3 h.shape: torch.Size([128, 8, 64])
DEBUG:root:4 h.shape: torch.Size([128, 8, 64])
DEBUG:root:5 h.shape: torch.Size([128, 8, 64])
DEBUG:root:6 h.shape: torch.Size([128, 8, 64])
DEBUG:root:7 h.shape: torch.Size([128, 8, 64])
DEBUG:root:8 h.shape: torch.Size([128, 8, 64])
DEBUG:root:9 h.shape: torch.Size([128, 8, 32])
DEBUG:root:10 h.shape: torch.Size([128, 256])
DEBUG:root:11 h.shape: torch.Size([128, 256])
DEBUG:root:12 h.shape: torch.Size([128, 1024])
DEBUG:root:13 h.shape: torch.Size([128, 1024])
DEBUG:root:14 h.shape: torch.Size([128, 1024])
DEBUG:root:15 h.shape: torch.Size([128, 1])

TimeDistributedReducedSum
DEBUG:root:8 h.shape: torch.Size([128, 8, 64])
  reducedSum
DEBUG:root:9 h.shape: torch.Size([128, 8]), h.size = <built-in method size of Tensor object at 0x786c597fad50>
  time distributed flatten
DEBUG:root:10 h.shape: torch.Size([128, 8]), h.size = <built-in method size of Tensor object at 0x786c597fad50>
  
  
  okay, so I want 2 variants: 
      1. [128, 8, 64] -> reducedSum -> [128, 8], 
      or 2. [128, 8, 64] -> reducedSum -> [128, 64]
"""
from torch import nn
import torch
import mycolors
from extensions import timber, create_conv_sequence


class Cnn1dClassifier(nn.Module):
  def __init__(self,
               seq_len,
               in_channel_num_of_nucleotides=4,
               kernel_size_k_mer_motif=4,
               num_filters=32,
               lstm_hidden_size=128,
               dnn_size=128,
               conv_seq_list_size=3,
               *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.seq_layer_forward = create_conv_sequence(in_channel_num_of_nucleotides, num_filters,
                                                  kernel_size_k_mer_motif)
    self.seq_layer_backward = create_conv_sequence(in_channel_num_of_nucleotides, num_filters,
                                                   kernel_size_k_mer_motif)

    self.flatten = nn.Flatten()

    dnn_in_features = int(num_filters * (seq_len * 2) / kernel_size_k_mer_motif)  # no idea why
    # two because forward_sequence,and backward_sequence
    self.dnn = nn.Linear(in_features=dnn_in_features, out_features=dnn_size)
    self.dnn_activation = nn.ReLU(inplace=True)
    self.dropout = nn.Dropout(p=0.33)

    self.output_layer = nn.Linear(in_features=dnn_size, out_features=1)
    self.output_activation = torch.sigmoid  # not needed if using nn.BCEWithLogitsLoss()
    pass

  def forward(self, x):
    xf, xb = x[0], x[1]

    hf = self.seq_layer_forward(xf)
    timber.debug(mycolors.red + f"1{ hf.shape = }")
    hb = self.seq_layer_backward(xb)
    timber.debug(mycolors.green + f"2{ hb.shape = }")

    h = torch.concatenate(tensors=(hf, hb), dim=2)
    timber.debug(mycolors.yellow + f"4{ h.shape = } concat")

    h = self.flatten(h)
    timber.debug(mycolors.yellow + f"5{ h.shape = } flatten")

    h = self.dnn(h)
    timber.debug(mycolors.yellow + f"8{ h.shape = } dnn")
    h = self.dnn_activation(h)
    timber.debug(mycolors.blue + f"9{ h.shape = } dnn_activation")
    h = self.dropout(h)
    timber.debug(mycolors.blue + f"10{ h.shape = } dropout")
    h = self.output_layer(h)
    timber.debug(mycolors.blue + f"11{ h.shape = } output_layer")
    h = self.output_activation(h)
    timber.debug(mycolors.blue + f"12{ h.shape = } output_activation")
    return h

"""
L1 regularization => everything 0.5, L2 regularization much better.
Hence using L2 regularization

seq len 100 -> acc .80, auc 0.90 (dnn_size = 128, k-mer-size = 4, num_filters = 32)  * the baseline. Improve it!
        200 -> acc .64, auc .71  (dnn_size = 128, 512)
               acc .71, auc .79  (dnn_size = 1024)
        400 -> acc .57, auc .63  (dnn_size = 1024)
        400 -> acc .63, auc .68  (lots of params changed)
        
        kmer_size 8 (ie, gt 4), num_filters > 32, dnn_size = 1024 leads to overfitting caused by too many parameters 
        that memorize both, a pattern and lot's of noise.
  
      1000 -> acc .49, auc .50 (dnn 128, num_filters = 32, k_mer_size = 4)
      
      Need to consolidate for seq len 400 or 1000
"""

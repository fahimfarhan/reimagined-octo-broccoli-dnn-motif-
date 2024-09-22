from matplotlib import pyplot as plt
from torch import nn
import torch

import torch.nn.functional as F

import mycolors
from extensions import timber, create_conv_sequence
from start import start


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
    self.file_name = f"weights_Cnn1dClassifier_seqlen_{seq_len}.pth"

    self.seq_layer_forward = create_conv_sequence(in_channel_num_of_nucleotides, num_filters,
                                                  kernel_size_k_mer_motif)
    self.seq_layer_backward = create_conv_sequence(in_channel_num_of_nucleotides, num_filters,
                                                   kernel_size_k_mer_motif)

    self.flatten = nn.Flatten()

    dnn_in_features = int(num_filters * (seq_len * 2) / kernel_size_k_mer_motif)  # no idea why
    # two because forward_sequence,and backward_sequence
    self.dnn = nn.Linear(in_features=dnn_in_features, out_features=dnn_size)
    self.dnn_activation = nn.ReLU(inplace=False)  # inplace = true messes with interpretability!
    self.dropout = nn.Dropout(p=0.33)

    self.output_layer = nn.Linear(in_features=dnn_size, out_features=1)
    self.output_activation = torch.sigmoid  # not needed if using nn.BCEWithLogitsLoss()

    self.enable_logging = False
    self.layer_output_logger: dict = {}
    pass

  def forward(self, x):
    xf, xb = x[0], x[1]

    hf = self.seq_layer_forward(xf)
    if self.enable_logging:
      self.layer_output_logger["seq_layer_forward"] = hf
    timber.debug(mycolors.red + f"1{ hf.shape = }")
    hb = self.seq_layer_backward(xb)
    if self.enable_logging:
      self.layer_output_logger["seq_layer_backward"] = hb
    timber.debug(mycolors.green + f"2{ hb.shape = }")

    h = torch.concatenate(tensors=(hf, hb), dim=2)
    timber.debug(mycolors.yellow + f"4{ h.shape = } concat")

    h = self.flatten(h)
    timber.debug(mycolors.yellow + f"5{ h.shape = } flatten")

    h = self.dnn(h)
    timber.debug(mycolors.yellow + f"8{ h.shape = } dnn")
    if self.enable_logging:
      self.layer_output_logger["dnn"] = h

    h = self.dnn_activation(h)
    if self.enable_logging:
      self.layer_output_logger["dnn_activation"] = h
    timber.debug(mycolors.blue + f"9{ h.shape = } dnn_activation")
    h = self.dropout(h)
    timber.debug(mycolors.blue + f"10{ h.shape = } dropout")
    h = self.output_layer(h)
    if self.enable_logging:
      self.layer_output_logger["output_layer"] = h
    timber.debug(mycolors.blue + f"11{ h.shape = } output_layer")
    h = self.output_activation(h)
    if self.enable_logging:
      self.layer_output_logger["output_activation"] = h
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


def visualize_layer_output_activations(layer_output, title="Activation Map"):
  # Assuming we want to visualize the activations of the first sample
  activation_map = layer_output.detach().cpu().numpy()[0]
  plt.imshow(activation_map, aspect='auto', cmap='hot')
  plt.colorbar()
  plt.title(title)
  plt.show()


if __name__ == '__main__':
  WINDOW = 4000
  simple_cnn = Cnn1dClassifier(seq_len=WINDOW)
  simple_cnn.enable_logging = True

  start(classifier_model=simple_cnn, model_save_path=simple_cnn.file_name, WINDOW=WINDOW,
        dataset_folder_prefix="inputdata/", is_debug=True)

  for key, value in simple_cnn.layer_output_logger.items():
    print(f"{key = }, f{value = }")
    try:
      visualize_layer_output_activations(value, title=key)
    except Exception as x:
      print(x)
  pass

"""
Results:

Debug == True:
        Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      test_accuracy         0.9752500057220459
        test_auc            0.9877760410308838
      test_f1_score         0.9757412672042847
        test_loss           0.10312975198030472
     test_precision         0.9567515850067139
       test_recall          0.9955000281333923

Debug == False:
    basically acc, auc etc == 50%
"""

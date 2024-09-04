from extensions import CommonAttentionLayer
from start import *


class Cnn1dAttentionClassifier(nn.Module):
  def __init__(self, 
               seq_len,
               # device,
               in_channel_num_of_nucleotides=4,
               kernel_size_k_mer_motif=4,
               num_filters=32,
               lstm_hidden_size=128,
               dnn_size=512,
               conv_seq_list_size=2,
               *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.file_name = f"weights_Cnn1dAttentionClassifier.pth"

    # CNN layers
    self.seq_layer_forward = create_conv_sequence(in_channel_num_of_nucleotides, num_filters,
                                                  kernel_size_k_mer_motif)
    self.seq_layer_backward = create_conv_sequence(in_channel_num_of_nucleotides, num_filters,
                                                   kernel_size_k_mer_motif)

    self.conv1 = create_conv_sequence(num_filters, 64, kernel_size_k_mer_motif) # nn.Conv1d(in_channels=num_filters, out_channels=64, kernel_size=3, padding=1)
    # self.conv2 = create_conv_sequence(64, 128, kernel_size_k_mer_motif)

    self.attention_layer = CommonAttentionLayer(hidden_size=12)

    self.dnn_layer = nn.Linear(in_features=12, out_features=1)
    self.output_activation = torch.sigmoid  # not needed if using nn.BCEWithLogitsLoss()

    pass

  def forward(self, x):
    h, ignore = self.forward_for_interpretation(x)
    return h

  def forward_for_interpretation(self, x):
    xf, xb = x[0], x[1]

    hf = self.seq_layer_forward(xf)
    timber.debug(mycolors.red + f"1{ hf.shape = }")
    hb = self.seq_layer_backward(xb)
    timber.debug(mycolors.green + f"2{ hb.shape = }")

    h = torch.concatenate(tensors=(hf, hb), dim=2)
    h = self.conv1(h)
    # h = self.conv2(h)
    context_vector, attention_weight = self.attention_layer(h)
    h = self.dnn_layer(context_vector)
    h = self.output_activation(h)
    return h, attention_weight

if __name__ == '__main__':
  window = 200
  attention_model = Cnn1dAttentionClassifier(window)
  start(attention_model, attention_model.file_name, is_attention_model=True, WINDOW=window, dataset_folder_prefix="inputdata/")
  pass

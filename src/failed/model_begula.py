import torch
import torch.nn as nn

from extensions import create_conv_sequence, timber
import mycolors
from src.start import start

"""
Begula code reference: https://github.com/kipoi/models/blob/master/DeepSEA/beluga/model.py
"""


class LambdaBase(nn.Sequential):
  def __init__(self, fn, *args):
    super(LambdaBase, self).__init__(*args)
    self.lambda_func = fn

  def forward_prepare(self, input):
    output = []
    for module in self._modules.values():
      output.append(module(input))
    return output if output else input


class Lambda(LambdaBase):
  def forward(self, input):
    return self.lambda_func(self.forward_prepare(input))


class Beluga(nn.Module):
  def __init__(self):
    super(Beluga, self).__init__()
    self.model = nn.Sequential(
      nn.Sequential(
        nn.Conv2d(4, 320, (1, 8)),
        nn.ReLU(),
        nn.Conv2d(320, 320, (1, 8)),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.MaxPool2d((1, 4), (1, 4)),
        nn.Conv2d(320, 480, (1, 8)),
        nn.ReLU(),
        nn.Conv2d(480, 480, (1, 8)),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.MaxPool2d((1, 4), (1, 4)),
        nn.Conv2d(480, 640, (1, 8)),
        nn.ReLU(),
        nn.Conv2d(640, 640, (1, 8)),
        nn.ReLU(),
      ),
      nn.Sequential(
        nn.Dropout(0.5),
        Lambda(lambda x: x.view(x.size(0), -1)),
        nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(67840, 2003)),
        nn.ReLU(),
        nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(2003, 2002)),
      ),
      nn.Sigmoid(),
    )

  def forward(self, x):
    return self.model(x)


"""
Beluga is having a size missmatch, will repair it later
"""


class BelugaMQTLClassifier(nn.Module):
  def __init__(self, seq_len, in_channel_num_of_nucleotides=4, kernel_size_k_mer_motif=1, num_filters=32,
               lstm_hidden_size=128, dnn_size=128, conv_seq_list_size=3, *args, **kwargs):
    # input layer
    super().__init__(*args, **kwargs)
    self.model_name = f"BelugaMQTLClassifier"

    self.seq_layer_forward = create_conv_sequence(in_channel_num_of_nucleotides, num_filters,
                                                  kernel_size_k_mer_motif)
    self.seq_layer_backward = create_conv_sequence(in_channel_num_of_nucleotides, num_filters,
                                                   kernel_size_k_mer_motif)

    self.hidden1 = create_conv_sequence(num_filters, 4, kernel_size_k_mer_motif)  # to match the dimensions with begula

    # begula model as a layer
    self.beluga = Beluga()
    # output layer
    self.output_layer = nn.Linear(2002, 1)
    self.output_activation = torch.sigmoid  # not needed if using nn.BCEWithLogitsLoss()

  def forward(self, x):
    xf, xb = x[0], x[1]

    hf = self.seq_layer_forward(xf)
    timber.debug(mycolors.red + f"1{ hf.shape = }")
    hb = self.seq_layer_backward(xb)
    timber.debug(mycolors.green + f"2{ hb.shape = }")

    h = torch.concatenate(tensors=(hf, hb), dim=2)
    timber.debug(mycolors.green + f"3{ h.shape = }")
    h = self.hidden1(h)
    timber.debug(mycolors.green + f"4{ h.shape = }")
    #  [1, 16, 4, 25]
    # h = h.permute(0, 2, 1)  # [1, 16, 25, 4]
    h = h.permute(1, 2, 0)

    # h = xf.permute(1, 2, 0)
    h = self.beluga(h)
    timber.debug(mycolors.green + f"5{ h.shape = } beluga output")

    h = self.output_layer(h)
    h = self.output_activation(h)

    return h


if __name__ == "__main__":
  # FAILED!!! SIZE Mismatch error!
  window = 200

  beluga_classifier = BelugaMQTLClassifier(seq_len=window)
  start(classifier_model=beluga_classifier,
        model_save_path=f"weights_{beluga_classifier.model_name}.pth",
        m_optimizer=torch.optim.RMSprop,
        WINDOW=window,
        dataset_folder_prefix="inputdata/"
        )
  pass

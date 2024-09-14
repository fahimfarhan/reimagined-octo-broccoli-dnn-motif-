import torch
import torch.nn as nn

from src.extensions import timber
from start import start


class LSTMDNAClassifier(nn.Module):
  def __init__(self,
               seq_len, hidden_size, num_layers, num_classes, dropout=0.1):
    super(LSTMDNAClassifier, self).__init__()

    self.file_name = f"weights_LSTMDNAClassifier_seqlen_{seq_len}.pth"


    # LSTM layer
    is_bidirectional = True
    number_of_nucleotides = 4
    self.lstm = nn.LSTM(input_size=number_of_nucleotides,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_first=True,
                        bidirectional=is_bidirectional,
                        dropout=dropout)

    # Fully connected layer for classification
    lstm_output_size = hidden_size
    if is_bidirectional:
      lstm_output_size = lstm_output_size * 2

    self.fc = nn.Linear(lstm_output_size, num_classes)

    # Dropout layer
    self.dropout = nn.Dropout(dropout)
    self.output_activation = torch.sigmoid  # not needed if using nn.BCEWithLogitsLoss()


  def forward(self, x):
    hf, hb = x[0], x[1]
    # hf.shape = [batch_size, num_of_nucleotides, dna_seq_len]
    # timber.info(f"{hf.shape = }")
    h = hf.permute(0, 2, 1)
    # timber.info(f"{h.shape = }")
    # after reshaping: h.shape = [batch_size, dna_seq_len, num_of_nucleotides]
    # LSTM forward pass
    lstm_out, (hn, cn) = self.lstm(h)  # lstm_out: (batch_size, seq_len, hidden_size)

    # Use the last hidden state for classification
    out = lstm_out[:, -1, :]  # Extract the last time step output

    # Apply dropout and the final fully connected layer
    out = self.dropout(out)
    out = self.fc(out)
    out = self.output_activation(out)
    return out


if __name__ == '__main__':
  window = 200
  simple_cnn = LSTMDNAClassifier(200, 128, 2, 1)
  start(classifier_model=simple_cnn, model_save_path=simple_cnn.file_name, WINDOW=window,
        dataset_folder_prefix="inputdata/", is_debug=True)
  pass

"""
simple lstm is completely incapable of detecting anything! worse than cnn?
What am I doing wrong?
is_debug = True,
acc, auc = 50%
"""
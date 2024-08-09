from src.models_simple_cnn_1d_tdf import SimpleCNN1dTdfClassifier
from start import *

if __name__ == '__main__':
  pytorch_model = SimpleCNN1dTdfClassifier(seq_len=WINDOW)
  start(classifier_model=pytorch_model, model_save_path=pytorch_model.file_name)
  pass

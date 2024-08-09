from start import *

if __name__ == '__main__':
  simple_cnn = Cnn1dClassifier(seq_len=WINDOW)
  start(classifier_model=simple_cnn, model_save_path=simple_cnn.file_name)
  pass

import random

import grelu
from grelu.lightning import LightningModel
from grelu.lightning import PatternMarginalizeDataset

from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from extensions import *
from models_cnn_1d import Cnn1dClassifier

# df = pd.read_csv("small_dataset.csv")
WINDOW = 100
DEBUG_MOTIF = "ATCGTTCA"
# LEN_DEBUG_MOTIF = 8
DEBUG = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resize_and_insert_motif_if_debug(seq: str, label: int) -> str:
  # else label is 1
  mid = int(len(seq) / 2)
  start = mid - int(WINDOW / 2)
  end = start + WINDOW

  if label == 0:
    return seq[start: end]

  if not DEBUG:
    return seq[start: end]

  rand_pos = random.randrange(start, (end - len(DEBUG_MOTIF)))
  random_end = rand_pos + len(DEBUG_MOTIF)
  output = seq[start: rand_pos] + DEBUG_MOTIF + seq[random_end: end]
  # print(f"{start = }, { rand_pos = }, { random_end = }, { end = }, { len(DEBUG_MOTIF) = }")
  assert len(output) == WINDOW
  return output


def get_dataframe(shuffled: bool = True) -> pd.DataFrame:
  df = pd.read_csv("small_dataset.csv")
  tmp = [resize_and_insert_motif_if_debug(seq=df["sequence"][idx], label=int(df["yes_mqtl"][idx])) for idx in
         df.index]  # todo fix this
  # timber.debug(tmp)
  df["sequence"] = tmp
  if not shuffled:
    return df
  shuffle_df = df.sample(frac=1)  # shuffle the dataframe

  return shuffle_df



class MqtlDataModule(LightningDataModule):
  def __init__(self, train_ds: MyDataSet, val_ds: MyDataSet, test_ds: MyDataSet, batch_size=16):
    super().__init__()
    self.batch_size = batch_size
    self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=15,
                                   persistent_workers=True)
    self.validate_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=15,
                                      persistent_workers=True)
    self.test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, num_workers=15,
                                  persistent_workers=True)
    pass

  def prepare_data(self):
    pass

  def setup(self, stage: str) -> None:
    timber.info(f"inside setup: {stage = }")
    pass

  def train_dataloader(self) -> TRAIN_DATALOADERS:
    return self.train_loader

  def val_dataloader(self) -> EVAL_DATALOADERS:
    return self.validate_loader

  def test_dataloader(self) -> EVAL_DATALOADERS:
    return self.test_loader

def start_failed():
  df: pd.DataFrame = get_dataframe()
  for seq in df["sequence"]:
    # print(f"{len(seq)}")
    assert (len(seq) == WINDOW)

  # experiment = 'tutorial_3'
  # if not os.path.exists(experiment):
  #   os.makedirs(experiment)

  x_train, x_tmp, y_train, y_tmp = train_test_split(df["sequence"], df["yes_mqtl"], test_size=0.2)
  x_test, x_val, y_test, y_val = train_test_split(x_tmp, y_tmp, test_size=0.5)

  train_dataset = MyDataSet(x_train, y_train)
  val_dataset = MyDataSet(x_val, y_val)
  test_dataset = MyDataSet(x_test, y_test)

  data_module = MqtlDataModule(train_ds=train_dataset, val_ds=val_dataset, test_ds=test_dataset)
  # classifier_model = SimpleCNN1DmQtlClassifier(seq_len=WINDOW)
  classifier_model = Cnn1dClassifier(seq_len=WINDOW)  # .double()
  # classifier_model = classifier_model.to(DEVICE)
  classifier_model_as_dict = model_to_dict(classifier_model)
  classifier_model_as_dict["model_type"] = "ConvModel"
  # print(classifier_model_as_dict)

  classifier_module = grelu.lightning.LightningModel(model_params=classifier_model_as_dict)

  trainer = Trainer(max_epochs=2, precision="32")
  trainer.fit(model=classifier_module, datamodule=data_module)
  timber.info("\n\n")
  trainer.test(model=classifier_module, datamodule=data_module)
  timber.info("\n\n")
  pass


def my_test():
  classifier_model = Cnn1dClassifier(seq_len=WINDOW)  # .double()
  # classifier_model = classifier_model.to(DEVICE)
  classifier_model_as_dict = model_to_dict(classifier_model)
  print(classifier_model_as_dict)
  pass

if __name__ == '__main__':
  start_failed()
  # my_test()
  pass

import time

import grelu.data.preprocess
import pandas as pd
from _00_constants import *

if __name__ == "__main__":
  start_time = time.time()

  df_unfiltered = pd.read_csv(f"dataset_{WINDOW}.csv")


  df = df_unfiltered.dropna(subset=["sequence"])
  df = df[df["sequence"].notnull()]
  df = df[df["sequence"] != ""]

  row = df.iloc[390]
  seq = row["sequence"]
  print(row)
  print(f"{seq = }")

  # print(df["sequence"])

  train, validate, test = grelu.data.preprocess.split(df)
  print(train.head())
  train.to_csv(f"dataset_{WINDOW}_train.csv")
  validate.to_csv(f"dataset_{WINDOW}_validate.csv")
  test.to_csv(f"dataset_{WINDOW}_test.csv")

  # Record the end time
  end_time = time.time()
  # Calculate the duration
  duration = end_time - start_time
  # Print the runtime
  print(f"Runtime: {duration:.2f} seconds")

  pass

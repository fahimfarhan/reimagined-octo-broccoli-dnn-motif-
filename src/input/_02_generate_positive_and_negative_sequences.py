import time

import pandas as pd
# import grelu as grelu
from grelu.data.preprocess import get_gc_matched_intervals
from _00_constants import *


def start():
  df_unfiltered = pd.read_csv(
    F"st5_filtered_cosmopolitan_meqtl_snp_cpg_distance_lte_{SLIGHTLY_LARGER_WINDOW}.txt",
    sep="\t",
    index_col=0
  )
  print(df_unfiltered.head())
  #based on this tutorial, I need: https://genentech.github.io/gReLU/tutorials/3_train.html
  positive_seqs: pd.DataFrame = pd.DataFrame()
  # df = df_unfiltered[df_unfiltered['snp.chr'] == 1]
  df = df_unfiltered
  positive_seqs["chrom"] = "chr" + (df["snp.chr"].astype(str))
  positive_seqs["start"] = (df["snp.pos"].astype(int) - HALF_WINDOW)
  positive_seqs["end"] = positive_seqs["start"] + WINDOW
  print("--------- positive head -------------")

  # positive_seqs = positive_seqs.head(100)
  print(positive_seqs.head())

  # creating negative dataset
  genome = "hg38"

  negatives = get_gc_matched_intervals(  # grelu.data.preprocess.get_gc_matched_intervals(
    positive_seqs,
    binwidth=0.02,  # resolution of measuring GC content
    genome=genome,
    chroms="autosomes",  # negative regions will also be chosen from autosomes
    # gc_bw_file='gc_hg38_2114.bw',
    blacklist=genome,  # negative regions overlapping the blacklist will be dropped
    seed=0,
  )
  print("--------- negative head -------------")
  print(negatives.head())
  positive_seqs.to_csv(f"positives_{WINDOW}.csv")
  negatives.to_csv(f"negatives_{WINDOW}.csv")
  pass


if __name__ == "__main__":
  # Record the start time
  start_time = time.time()
  # run code
  start()
  # Record the end time
  end_time = time.time()
  # Calculate the duration
  duration = end_time - start_time
  # Print the runtime
  print(f"Runtime: {duration:.2f} seconds")
  pass

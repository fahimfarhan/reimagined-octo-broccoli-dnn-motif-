import pandas as pd
# import grelu as grelu
from grelu.data.preprocess import get_gc_matched_intervals


if __name__ == "__main__":
  window = 1000
  half_window = 500

  df = pd.read_csv(
    "st5_filtered_cosmopolitan_meqtl_snp_cpg_distance_lte_5000.txt",
    sep="\t",
    index_col=0
  )
  print(df.head())
  #based on this tutorial, I need: https://genentech.github.io/gReLU/tutorials/3_train.html
  positive_seqs: pd.DataFrame = pd.DataFrame()
  positive_seqs["chrom"] = "chr"+(df["snp.chr"].astype(str))
  positive_seqs["start"] = (df["snp.pos"].astype(int) - half_window)
  positive_seqs["end"] = positive_seqs["start"] + window
  print("--------- positive head -------------")

  positive_seqs = positive_seqs.head(100)
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

  pass

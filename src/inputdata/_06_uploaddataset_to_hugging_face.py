from datasets import load_dataset, DatasetDict

if __name__ == "__main__":
  data_files = {
    "train": "dataset_200_train_binned.csv",
    "validate": "dataset_200_validate_binned.csv",
    "test": "dataset_200_test_binned.csv",
  }
  dataset = load_dataset("csv", data_files=data_files)

  # Push dataset to Hugging Face hub
  dataset.push_to_hub("fahimfarhan/mqtl-classification-dataset-binned-200")
  pass

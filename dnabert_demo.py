import torch
from transformers import AutoTokenizer, AutoModel


if __name__ == "__main__":
    pretrained_model_name = "zhihan1996/DNA_bert_6"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(pretrained_model_name, trust_remote_code=True)

    dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
    inputs = tokenizer(dna, return_tensors='pt')["input_ids"]
    hidden_states = model(inputs)[0]  # [1, sequence_length, 768]

    # embedding with mean pooling
    embedding_mean = torch.mean(hidden_states[0], dim=0)
    print(embedding_mean.shape)  # expect to be 768

    # embedding with max pooling
    embedding_max = torch.max(hidden_states[0], dim=0)[0]
    print(embedding_max.shape)  # expect to be 768

    pass


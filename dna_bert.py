import torch
from transformers import BertTokenizer, BertModel, BatchEncoding
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

# load the DNABert tokenizer, and model
dna_bert_6 = "zhihan1996/DNA_bert_6"  # works on my laptop!

bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=dna_bert_6)
bert_model = BertModel.from_pretrained(pretrained_model_name_or_path=dna_bert_6)


def preprocess_dna(sequence, k=6):
    tokens = [sequence[i:i + k] for i in range(len(sequence) - k + 1)]
    return tokens


def run1():
    dna_sequence = "ATCGTAGCTAGCTAGCTGACT"
    tokens: list[str] = preprocess_dna(dna_sequence)
    print(f"{tokens = }")
    print(f"typeoftokens = {type(tokens) = }")
    encoded_input: BatchEncoding = bert_tokenizer(tokens, return_tensors="pt", is_split_into_words=True)
    print(f"{encoded_input = }")
    print(f"{type(encoded_input) = }")
    """
    tokens = ['ATCGTA', 'TCGTAG', 'CGTAGC', 'GTAGCT', 'TAGCTA', 'AGCTAG', 'GCTAGC', 'CTAGCT', 'TAGCTA', 'AGCTAG',
              'GCTAGC', 'CTAGCT', 'TAGCTG', 'AGCTGA', 'GCTGAC', 'CTGACT']
    typeoftokens = type(tokens) = <

    class 'list'>

    encoded_input = {'input_ids': tensor([[2, 441, 1752, 2899, 3390, 1257, 920, 3667, 2366, 1257, 920, 3667,
                                           2366, 1260, 929, 3703, 2510, 3]]),
                     'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                     'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
    type(encoded_input) = <

    class 'transformers.tokenization_utils_base.BatchEncoding'>
    """

    bert_outputs: BaseModelOutputWithPoolingAndCrossAttentions = bert_model(**encoded_input)
    """
    BaseModelOutputWithPoolingAndCrossAttentions = { 
        last_hidden_state: tensor, 
        pooler_output: tensor, 
        grad_fn, 
        hidden_states,
        past_key_values,
        attentions,
        cross_attentions
    }
    """
    print(f"{bert_outputs = }")
    print(f"{type(bert_outputs) = }")


if __name__ == '__main__':
    run1()

from transformers import BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from start import *


class ReshapedBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
  def forward(self, input, target):
    return super().forward(input.squeeze(), target.float())


class MQtlDnaBERT6Classifier(nn.Module):
  def __init__(self,
               bert_model=BertModel.from_pretrained(pretrained_model_name_or_path=DNA_BERT_6),
               hidden_size=768,
               num_classes=1,
               *args,
               **kwargs
               ):
    super().__init__(*args, **kwargs)

    self.model_name = "MQtlDnaBERT6Classifier"

    self.bert_model = bert_model
    self.attention = CommonAttentionLayer(hidden_size)
    self.classifier = nn.Linear(hidden_size, num_classes)
    pass

  def forwardV1Failed(self, encoded_input_x: BatchEncoding):
    input_ids: torch.tensor = encoded_input_x["input_ids"]
    token_type_ids: torch.tensor = encoded_input_x["token_type_ids"]
    attention_mask: torch.tensor = encoded_input_x["attention_mask"]
    """
       bert_output: BaseModelOutputWithPoolingAndCrossAttentions = self.bert_model(
         input_ids=input_ids,
         attention_mask=attention_mask,
         token_type_ids=token_type_ids
       )
       """
    # looks like equivalent to the next line...
    bert_output: BaseModelOutputWithPoolingAndCrossAttentions = self.bert_model(**encoded_input_x)

    last_hidden_state = bert_output.last_hidden_state
    context_vector = self.attention(last_hidden_state)
    y = self.classifier(context_vector)
    return y

  def forwardV2Ok(self, input_ids: torch.tensor, attention_mask: torch.tensor, token_type_ids):
    # input_ids: torch.tensor = encoded_input_x["input_ids"]
    # token_type_ids: torch.tensor = encoded_input_x["token_type_ids"]
    # attention_mask: torch.tensor = encoded_input_x["attention_mask"]
    """
       bert_output: BaseModelOutputWithPoolingAndCrossAttentions = self.bert_model(
         input_ids=input_ids,
         attention_mask=attention_mask,
         token_type_ids=token_type_ids
       )
       """
    # looks like equivalent to the next line...
    inputs_embeds = None
    if input_ids is not None and inputs_embeds is not None:
      raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
      input_shape = input_ids.size()
      print(f"1 { input_shape = }")
    elif inputs_embeds is not None:
      input_shape = inputs_embeds.size()[:-1]
      print(f"2 { input_shape = }")
    else:
      raise ValueError("You have to specify either input_ids or inputs_embeds")

    # torch.Size([128, 1, 512])
    print(f"3 {input_ids = }")
    input_ids = input_ids.squeeze(dim=1).to(DEVICE)
    print(f"4 {input_ids}")

    print(f"5 attention shape: {attention_mask.shape = }")
    print(f"5 attention size: {attention_mask.size = }")

    attention_mask = attention_mask.squeeze(dim=1).to(DEVICE)

    print(f"5 token_type_ids shape: {token_type_ids.shape = }")
    print(f"5 token_type_ids size: {token_type_ids.size = }")

    token_type_ids = token_type_ids.squeeze(dim=1).to(DEVICE)

    bert_output: BaseModelOutputWithPoolingAndCrossAttentions = self.bert_model(
      input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids
    )

    last_hidden_state = bert_output.last_hidden_state
    context_vector = self.attention(last_hidden_state)
    y = self.classifier(context_vector)
    return y

  def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, token_type_ids):
    """
    # torch.Size([128, 1, 512]) --> [128, 512]
    input_ids = input_ids.squeeze(dim=1).to(DEVICE)
    # torch.Size([16, 1, 512]) --> [16, 512]
    attention_mask = attention_mask.squeeze(dim=1).to(DEVICE)
    token_type_ids = token_type_ids.squeeze(dim=1).to(DEVICE)
    """
    bert_output: BaseModelOutputWithPoolingAndCrossAttentions = self.bert_model(
      input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids
    )

    last_hidden_state = bert_output.last_hidden_state
    context_vector, ignore_attention_weight = self.attention(last_hidden_state)
    y = self.classifier(context_vector)
    return y


if __name__ == "__main__":
  pytorch_model = MQtlDnaBERT6Classifier()
  start_bert(classifier_model=pytorch_model, model_save_path=f"weights_{pytorch_model.model_name}.pth",
             criterion=ReshapedBCEWithLogitsLoss(), WINDOW=200)
  pass

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer



class BERTEncoder(nn.Module):
    def __init__(self, model_name_or_path='bert-base-chinese', config=None, gradient_checkpointing=True):
        super().__init__()
        self.sentence_bert = AutoModel.from_pretrained(model_name_or_path, config)
        self.hidden_dim = self.sentence_bert.config.hidden_size
        self.sentence_bert.config.gradient_checkpointing = True  # important!
        print("initialized BERTEncoder")
        print(f"config.gradient_checkpointing = {self.sentence_bert.config.gradient_checkpointing}")

    # def forward(self, inputs):
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        :param input_ids: (batch_size, max_num_nodes, max_num_token,), here, tokens are tokenized by BERT tokenizer and ids are get
        :param attention_mask: ()
        :return:
        """
        # outputs = self.sentence_bert(**inputs)
        # print(input_ids.shape)
        # one instance should become one batch to input into PLM model
        batch_size, num_of_paragraph, max_seq_length = input_ids.shape
        if attention_mask == None:
            attention_mask = torch.ones_like(input_ids)
        input_ids = torch.reshape(input_ids, (-1, max_seq_length))
        attention_mask = torch.reshape(attention_mask, (-1, max_seq_length))

        outputs = self.sentence_bert(input_ids, attention_mask=attention_mask,)
        last_hidden_state = outputs.last_hidden_state
        cls_hidden = last_hidden_state[:,0,:]
        pooler_output = outputs.pooler_output
        # return cls_hidden

        # recover the batch dim
        pooler_output = torch.reshape(pooler_output, (batch_size, num_of_paragraph, -1))

        return pooler_output
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# hfl/chinese-xlnet-base
# hfl/chinese-xlnet-mid
# schen/longformer-chinese-base-4096
# Lowin/chinese-bigbird-wwm-base-4096
# Lowin/chinese-bigbird-base-4096  # need jieba tokenizer
# Lowin/chinese-bigbird-tiny-1024  # need jieba tokenizer
# Lowin/chinese-bigbird-small-1024  # need jieba tokenizer
# Lowin/chinese-bigbird-mini-1024  # need jieba tokenizer

class ConcatDocumentEncoder(nn.Module):
    def __init__(self, model_name_or_path='hfl/chinese-xlnet-base', config=None, gradient_checkpointing=True):
        super().__init__()
        # self.sentence_bert = AutoModel.from_pretrained(model_name_or_path, config)
        self.pretrained_model = AutoModel.from_pretrained(model_name_or_path)#, config)
        self.hidden_dim = self.pretrained_model.config.hidden_size
        self.pretrained_model.config.gradient_checkpointing = True  # important!
        print("initialized ConcatDocumentEncoder")
        print(f"config.gradient_checkpointing = {self.pretrained_model.config.gradient_checkpointing}")

    
    def __fetch_sep_rep(self, ten_output, seq_index):
        # copied from DAMT
        batch, _, _ = ten_output.shape
        sep_re_list = []
        for index in range(batch):
            cur_seq_index = seq_index[index]
            cur_output = ten_output[index]
            sep_re_list.append(cur_output[cur_seq_index])
        return torch.cat(sep_re_list, dim=0)

    def padding_sep_index_list(self, sep_index_list):
        # copied from DAMT
        """
        sep_index_list: List[List[int]]
        """
        max_edu = max([len(a) for a in sep_index_list])
        total_new_sep_index_list = []
        for index_list in sep_index_list:
            new_sep_index_list = []
            gap = max_edu - len(index_list)
            new_sep_index_list.extend(index_list)
            for i in range(gap):
                new_sep_index_list.append(index_list[-1])
            total_new_sep_index_list.append(new_sep_index_list)
        return max_edu, total_new_sep_index_list
    
    # def forward(self, inputs):
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        :param tokenized_document: (batch_size, max_num_token,)
        :param sep_index_list: List[List[int]]
        :param attention_mask: ()
        :return:
        """
        # tokenized_document = kwargs["tokenized_document"]  # (batch_size, max_num_token,)
        sep_index_list = kwargs["sep_index_list"]  # List[List[int]], without padding

        batch_size = input_ids.shape[0]
        edu_num, pad_sep_index_list = self.padding_sep_index_list(sep_index_list)
        # node_num = edu_num + 1

        outputs = self.pretrained_model(input_ids, attention_mask=attention_mask,)
        last_hidden_state = outputs.last_hidden_state
        hidden_vectors_sep = self.__fetch_sep_rep(last_hidden_state, pad_sep_index_list)
        hidden_vectors_sep = hidden_vectors_sep.reshape(batch_size, edu_num, -1)
        return hidden_vectors_sep


        """# outputs = self.sentence_bert(**inputs)
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

        return pooler_output"""
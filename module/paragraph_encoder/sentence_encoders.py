import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
# from sentence_transformers import SentenceTransformer


sentence_encoder_model_name_list = [
    'cyclone/simcse-chinese-roberta-wwm-ext',  # 768
    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',  # 768
]
# model_name='cyclone/simcse-chinese-roberta-wwm-ext'
# model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
# class SentenceEncoder(nn.Module):
#     def __init__(self, model_name='cyclone/simcse-chinese-roberta-wwm-ext'):
#         super().__init__()
#         # self.model = SentenceTransformer(model_name)
#         # self.encoder = SentenceTransformer(model_name)
#         self.encoder = None
#         self.config = AutoConfig.from_pretrained(model_name)
#         self.hidden_dim = config.hidden_size

#     def forward(self, sentences):
#         """

#         :param sentences: List[str]
#         :return:
#         """
#         # 11/28
#         sentence_embeddings = self.encoder.encode(sentences)
#         sentence_embeddings = torch.tensor(sentence_embeddings).to(self.device)
#         return sentence_embeddings

class SentenceEncoder(nn.Module):
    def __init__(self, model_name='cyclone/simcse-chinese-roberta-wwm-ext', config=None):
        super().__init__()
        # self.model = SentenceTransformer(model_name)
        # self.encoder = SentenceTransformer(model_name)
        # self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config)
        self.encoder.config.gradient_checkpointing = True  # important!
        self.hidden_dim = self.encoder.config.hidden_size
        print("initialized SentenceEncoder (SimCSE, SentenceBERT, etc.)")
        print(f"config.gradient_checkpointing = {self.encoder.config.gradient_checkpointing}")


    # def forward(self, encoded_input):
    def forward(self, input_ids, attention_mask, **kwargs):
        """

        :param encoded_input: {'input_ids', 'token_type_ids', 'attention_mask'}
        :return:
        """
        # encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        # 11/28

        batch_size, num_of_paragraph, max_seq_length = input_ids.shape
        if attention_mask == None:
            attention_mask = torch.ones_like(input_ids)
        input_ids = torch.reshape(input_ids, (-1, max_seq_length))
        attention_mask = torch.reshape(attention_mask, (-1, max_seq_length))

        model_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = mean_pooling(model_output, attention_mask)

        sentence_embeddings = torch.reshape(sentence_embeddings, (batch_size, num_of_paragraph, -1))
        return sentence_embeddings


if __name__ == "__main__":
    model_name='cyclone/simcse-chinese-roberta-wwm-ext'
    # model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

    # Sentences we want sentence embeddings for
    sentences = ['This is an example sentence', 'Each sentence is converted', '最新通报！广东新增11例本土确诊病例！高考在即，江门发出倡议！']

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    print(model.config.hidden_size)

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    print(encoded_input)

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    print(model_output)

    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    print("Sentence embeddings:", sentence_embeddings)
    print(sentence_embeddings.shape)
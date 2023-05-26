import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from module import BasicEmbeddingLayer


class XPathFFNN(nn.Module):
    def __init__(self, html_tag_vocab, node_embedding_dim=64, index_embedding_dim=64, max_xpath_length=50, hidden_dim=512, out_dim=64, use_index_information=True, aggregation_method="add", max_index_value=101):
        """
        MArkupLM https://aclanthology.org/2022.acl-long.420.pdf
        used when args.xpath_encoder_type = "ffnn"
        html_tag_vocab is need 
        :param node_embedding_dim = args.xpath_node_embedding_dim
        :param index_embedding_dim = args.xpath_index_embedding_dim
        :param use_index_information=True
        :param aggregation_method="add"
        """
        super().__init__()

        self.html_tag_embedding = BasicEmbeddingLayer(vocab=html_tag_vocab, embedding_dim=node_embedding_dim)
        self.xpath_index_embedding = nn.Embedding(max_index_value, index_embedding_dim)


        self.use_index_information = use_index_information
        self.aggregation_method = aggregation_method
        self.max_xpath_length = max_xpath_length
        
        single_tag_in_dim = node_embedding_dim
        if use_index_information:
            if aggregation_method == "add":
                assert node_embedding_dim == index_embedding_dim
                single_tag_in_dim = node_embedding_dim
            elif aggregation_method == "concate":
                single_tag_in_dim = node_embedding_dim + index_embedding_dim
        in_dim = single_tag_in_dim * max_xpath_length
        

        self.ffnn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        # self.hidden_dim = 2 * hidden_dim  # bidirectional
        self.single_tag_input_dim = single_tag_in_dim
        self.linear_layer_input_dim = in_dim
        self.embedding_dim = out_dim

        print("initialized XPathFFNN")

    def forward(self, xpath_tag_ids, xpath_index=None, xpath_length=None):
        """

        :param xpath_tag_ids: (batch_size, max_num_nodes, max_xpath_length) #, in_dim=token_embedding_size)
        :param xpath_index: (batch_size, max_num_nodes, max_xpath_length) #, in_dim=token_embedding_size)
        :param xpath_length: (batch_size, max_num_nodes), indicating the number of tokens in each xpath sequence. for padding NODE, num_token can be any int value <= max_num_token is OK
        :return:
        """
        batch_size, max_num_nodes, max_xpath_length = xpath_tag_ids.shape
        
        
        # encoding for each node, independent of document global information, so let batch_size * max_num_nodes to be a new batch
        
        if xpath_index is not None:
            assert xpath_tag_ids.shape == xpath_index.shape, (xpath_tag_ids.shape, xpath_index.shape)
            xpath_index = xpath_index.reshape(batch_size*max_num_nodes, max_xpath_length)
        xpath_tag_ids = xpath_tag_ids.reshape(batch_size*max_num_nodes, max_xpath_length)
        
        if xpath_length is None:
            xpath_length = torch.tensor([max_xpath_length]*(batch_size*max_num_nodes), dtype=torch.long)
        if type(xpath_length) is torch.Tensor:
            xpath_length = xpath_length.reshape(batch_size*max_num_nodes).to("cpu")
        else:
            xpath_length = torch.tensor(xpath_length).reshape(batch_size*max_num_nodes).to("cpu")

        inputs = self.html_tag_embedding(xpath_tag_ids)  # (batch_size*max_num_nodes, max_xpath_length, node_embedding_dim)
        if self.use_index_information:
            xpath_index_embeddings = self.xpath_index_embedding(xpath_index)  # (batch_size*max_num_nodes, max_xpath_length, index_embedding_dim)
            if self.aggregation_method == "add":
                inputs = inputs + xpath_index_embeddings
            elif self.aggregation_method == "concate":
                inputs = torch.cat((inputs, xpath_index_embeddings), dim=-1)
        
        total_num_nodes, max_xpath_length, input_embedding_dim = inputs.shape
        inputs = inputs.reshape(total_num_nodes, -1)  # concate each htmltag embedding in a xpath
        if max_xpath_length < self.max_xpath_length:  # padding zeros
            inputs = torch.cat((inputs, torch.zeros(total_num_nodes, (self.max_xpath_length-max_xpath_length)*self.single_tag_input_dim).to(inputs.device)), dim=-1)
        # inputs = inputs[:,:self.max_xpath_length]  # (total_num_nodes, self.max_xpath_length)
        inputs = inputs[:,:self.linear_layer_input_dim]
        
        
        sentence_encodings = self.ffnn(inputs)  # (total_num_nodes, out_dim)
        sentence_encodings = sentence_encodings.reshape(batch_size, max_num_nodes, -1)  # (batch_size, max_num_nodes, out_dim)
        
        return sentence_encodings
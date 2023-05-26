import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from module import BasicEmbeddingLayer

MODULE_DICT = {
    "rnn": nn.RNN,
    "lstm": nn.LSTM,
    "gru": nn.GRU,
}

class XPathLSTM(nn.Module):
    def __init__(self, html_tag_vocab, model_type="gru", node_embedding_dim=64, index_embedding_dim=64, hidden_dim=64, out_dim=64, num_layers=1, use_index_information=True, aggregation_method="add", max_index_value=101):
        """
        That is exactly the same as module/paragraph_encoder/rnn_encoder.py
        used when args.xpath_encoder_type = "rnn"
        html_tag_vocab is need 
        :param node_embedding_dim = args.xpath_node_embedding_dim
        :param index_embedding_dim = args.xpath_index_embedding_dim
        :param use_index_information=True
        :param aggregation_method="add"
        """
        super().__init__()

        self.model_type = model_type

        self.html_tag_embedding = BasicEmbeddingLayer(vocab=html_tag_vocab, embedding_dim=node_embedding_dim)
        self.xpath_index_embedding = nn.Embedding(max_index_value, index_embedding_dim)


        self.use_index_information = use_index_information
        self.aggregation_method = aggregation_method
        
        in_dim = node_embedding_dim
        if use_index_information:
            if aggregation_method == "add":
                assert node_embedding_dim == index_embedding_dim
                in_dim = node_embedding_dim
            elif aggregation_method == "concate":
                in_dim = node_embedding_dim + index_embedding_dim
        
        self.sentence_lstm = MODULE_DICT[model_type](
            input_size=in_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True,
            #proj_size=out_dim,
        )
        self.hidden_dim = 2 * hidden_dim  # bidirectional
        self.embedding_dim = self.hidden_dim

        print("initialized XPathLSTM")

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
        # print(xpath_length)
        else:
            xpath_length = torch.tensor(xpath_length).reshape(batch_size*max_num_nodes).to("cpu")

        # print(xpath_tag_ids)
        inputs = self.html_tag_embedding(xpath_tag_ids)
        # print(inputs)
        # print(inputs.shape)
        if self.use_index_information:
            xpath_index_embeddings = self.xpath_index_embedding(xpath_index)
            # print(xpath_index_embeddings.shape)
            if self.aggregation_method == "add":
                # print("add")
                inputs = inputs + xpath_index_embeddings
            elif self.aggregation_method == "concate":
                # print("concate")
                inputs = torch.cat((inputs, xpath_index_embeddings), dim=-1)
        
        # print(inputs)
        # print(xpath_length)
        packed_inputs = pack_padded_sequence(inputs, xpath_length, batch_first=True, enforce_sorted=False)

        if self.model_type=="lstm":
            # output, (h_n, c_n) = self.sentence_lstm(inputs)  # output: (batch_size, num_token, out_dim*bidirectional)
            output, (h_n, c_n) = self.sentence_lstm(packed_inputs)
        else:
            # output, h_n = self.sentence_lstm(inputs)
            output, h_n = self.sentence_lstm(packed_inputs)
        '''
        Outputs: output, (h_n, c_n)
        output: (batch_size, num_token, out_dim*bidirectional) when batch_first=True, output features (h_t) from the last layer of the LSTM, for each t
        h_n: (bidirectional*num_layers, batch_size, hidden_dim), the final hidden state for each element in the sequence
            When bidirectional=True, h_n will contain a concatenation of the final forward and reverse hidden states, respectively
        c_n is the same as h_n except that it is the final cell state for each element in the sequence
        '''
        forward_direction_last_token_hidden = h_n[-2,:,:]  # (batch_size * max_num_nodes, hidden_dim)
        backward_direction_last_token_hidden = h_n[-1,:,:]  # (batch_size * max_num_nodes, hidden_dim)
        # print(h_n.shape)
        # print(forward_direction_last_token_hidden.shape)
        # print(backward_direction_last_token_hidden.shape)
        sentence_encodings = torch.concat((forward_direction_last_token_hidden, backward_direction_last_token_hidden), dim=-1)  # (batch_size * max_num_nodes, 2*hidden_dim)
        sentence_encodings = sentence_encodings.reshape(batch_size, max_num_nodes, -1)  # (batch_size, max_num_nodes, 2*hidden_dim)

        return sentence_encodings
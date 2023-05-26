import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

MODULE_DICT = {
    "rnn": nn.RNN,
    "lstm": nn.LSTM,
    "gru": nn.GRU,
}

class RNNEncoder(nn.Module):
    def __init__(self, model_type="gru", in_dim=300, hidden_dim=512, out_dim=512, num_layers=1):
        """
        a sentence-level local encoder, relized by bi-RNNs,
        using concatnation of the first token hidden vector of the backward calculation + 
        the last token hidden vector of the forward calculation
        """
        super().__init__()
        self.sentence_lstm = MODULE_DICT[model_type](
            input_size=in_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True,
            proj_size=out_dim,
        )
        self.hidden_dim = 2 * hidden_dim  # bidirectional
        print("initialized RNNEncoder")

    def forward(self, inputs, num_tokens, **kwargs):
        """

        :param inputs: (batch_size, max_num_nodes, max_num_token, in_dim=token_embedding_size)
        :param num_tokens: (batch_size, max_num_nodes), indicating the number of tokens in each node. for padding NODE, num_token can be any int value <= max_num_token is OK
        :return:
        """
        batch_size, max_num_nodes, max_num_token, embedding_size = inputs.shape
        # encoding for each node, independent of document global information, so let batch_size * max_num_nodes to be a new batch
        inputs = inputs.reshape(batch_size*max_num_nodes, max_num_token, embedding_size)
        num_tokens = num_tokens.reshape(batch_size*max_num_nodes).to("cpu")

        packed_inputs = pack_padded_sequence(inputs, num_tokens, batch_first=True, enforce_sorted=False)

        if model_type=="lstm":
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
        sentence_encodings = torch.concat((forward_direction_last_token_hidden, backward_direction_last_token_hidden), dim=-1)  # (batch_size * max_num_nodes, 2*hidden_dim)
        sentence_encodings = sentence_encodings.reshape(batch_size, max_num_nodes, -1)  # (batch_size, max_num_nodes, 2*hidden_dim)

        return sentence_encodings
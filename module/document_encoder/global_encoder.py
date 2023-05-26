import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

MODULE_DICT = {
    "rnn": nn.RNN,
    "lstm": nn.LSTM,
    "gru": nn.GRU,
}

class DocumentLevelRNNEncoder(nn.Module):
    def __init__(self, model_type="gru", in_dim=768, hidden_dim=768, out_dim=768, num_layers=1, bidirectional=True):
        """
        an RNN implementation of document-level global paragraph encoder
        """
        super().__init__()
        self.document_lstm = MODULE_DICT[model_type](
            input_size=in_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional,
            # proj_size=out_dim,
        )
        # raise ValueError("proj_size argument is only supported for LSTM, not RNN or GRU")
        self.hidden_dim = 2 * hidden_dim if bidirectional else hidden_dim
        print("initialized DocumentLevelRNNEncoder")

    def forward(self, inputs, num_nodes):
        """

        :param inputs: (batch_size, max_num_nodes, in_dim)
        :param num_nodes: (batch_size), oringinal number of nodes in each document in the batch
        :return:
        """
        if type(num_nodes) is torch.Tensor:
            num_nodes = num_nodes.to("cpu")
            # num_nodes = torch.tensor(num_nodes, dtype=torch.int64)
            num_nodes = num_nodes.to(torch.int64)
        packed_inputs = pack_padded_sequence(inputs, num_nodes, batch_first=True, enforce_sorted=False)
        # output, _ = self.document_lstm(inputs)  # output: (batch_size, num_token, out_dim*bidirectional)
        output, _ = self.document_lstm(packed_inputs)
        '''
        Outputs: output, (h_n, c_n)
        output: (batch_size, num_token, out_dim*bidirectional) when batch_first=True, output features (h_t) from the last layer of the LSTM, for each t
        h_n: (bidirectional*num_layers, batch_size, hidden_dim), the final hidden state for each element in the sequence
            When bidirectional=True, h_n will contain a concatenation of the final forward and reverse hidden states, respectively
        c_n is the same as h_n except that it is the final cell state for each element in the sequence
        '''
        output = pad_packed_sequence(output,  batch_first=True)  # (batch_size, max_num_nodes, 2*hidden_dim)
        # print(output)
        return output[0]
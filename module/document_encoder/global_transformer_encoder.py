import math

import torch
import torch.nn as nn
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..position_encoding.positional_encoding import PositionalEncoding

# 12/26
class DocumentLevelTransformerEncoder(nn.Module):
    def __init__(self, in_dim=768, out_dim=768, dim_feedforward=2048, nhead=4, num_layers=6, dropout=0.1):
        """
        an Transformer implementation of document-level global paragraph encoder
        """
        super().__init__()
        # self.document_lstm = MODULE_DICT[model_type](
        #     input_size=in_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional,
        #     # proj_size=out_dim,
        # )
        self.pos_encoder = PositionalEncoding(d_model=in_dim, dropout=dropout)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=in_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(in_dim, out_dim)
        
        self.d_model = in_dim
        self.hidden_dim = out_dim

        self.init_weights()

        print("initialized DocumentLevelTransformerEncoder")

    def init_weights(self) -> None:
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
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
        
        
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        src_mask (Optional[Tensor]) – the additive mask for the src sequence (optional).
        src_mask: (S,S) or (N⋅num_heads,S,S).

        src_key_padding_mask (Optional[Tensor]) – the ByteTensor mask for src keys per batch (optional).
        src_key_padding_mask: (S) for unbatched input otherwise (N,S).

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        padding_mask = []
        # max_node_num = num_nodes.max().tolist()
        batch_size, max_num_nodes, in_dim = inputs.shape
        for num in num_nodes:
            padding_mask.append([0]*num + [1]*(max_num_nodes-num))  # 注意这里的mask和PLM的mask是反的，这里被mask的地方是1，正常的地方是0
        padding_mask = torch.BoolTensor(padding_mask).to(inputs.device)

        inputs = inputs * math.sqrt(self.d_model)
        inputs = self.pos_encoder(inputs)
        output = self.transformer_encoder(inputs, src_key_padding_mask=padding_mask)
        output = self.decoder(output)
        return output
# -*- coding: utf-8 -*-
import torch
from torch import nn, relu
# from margin_loss import MarginLoss
from .iter_lstm import IterativeLSTM


GLOBAL_MODULES = {"lstm": IterativeLSTM}

class PairwisePointerNetwork(nn.Module):
    def __init__(self, 
        input_dims=512, 
        loss_type="margin", 
        #layer_type="bilinear", 
        max_paragraph_num=200, 
        position_encoding_type=None, 
        position_encoding_dim=0,
        position_combine_method=None,
        relative_position_encoding_type="random",
        # relative_position_encoding_dim=None,
        relative_position_encoding_dim=200,
        relative_position_combine_method="bilinear",
        bilinear_combine="tail",
        global_sequence_module="lstm",
    ):
        super().__init__()
        # print("Use New Version")

        self.input_dims = input_dims
        self.paragraph_vector_dim = self.input_dims
        self.max_paragraph_num = max_paragraph_num

        self.position_encoding = None
        self.position_encoding_dim = None
        self.position_combine_mlp = None
        self.position_encoding_combine_method = None
        self._init_position_encoding(position_encoding_type=position_encoding_type, position_encoding_dim=position_encoding_dim)
        self._init_position_combine_method(combine_method=position_combine_method,)

        self.relative_position_encoding = None
        self.relative_position_encoding_dim = None

        self.pairwise_mlp_interaction = None
        self.pairwise_linear_interaction = None
        self.pairwise_bilinear_interaction = None
        self.local_pairwise_combine_method = None
        self._init_relative_position_encoding(position_encoding_type=relative_position_encoding_type, position_encoding_dim=relative_position_encoding_dim)
        self._init_local_pairwise_combine_method(combine_method=relative_position_combine_method, bilinear_combine=bilinear_combine)

        
        self.head_mlp = nn.Sequential(nn.Linear(input_dims, input_dims),
                                      nn.ReLU(),
                                      nn.Linear(input_dims, input_dims),
                                      nn.Tanh()
                                      )
        self.tail_mlp = nn.Sequential(nn.Linear(input_dims, input_dims),
                                      nn.ReLU(),
                                      nn.Linear(input_dims, input_dims),
                                      nn.Tanh()
                                      )
        
        self.use_global_sequence_module = False
        print(global_sequence_module)
        if global_sequence_module in GLOBAL_MODULES.keys():
            print("*"*100)
            self.use_global_sequence_module = True
            self.iterlstm = GLOBAL_MODULES[global_sequence_module](input_dim=self.paragraph_vector_dim, hidden_dim=self.paragraph_vector_dim)
            self.score_mlp = nn.Sequential(
                nn.Linear(2*self.paragraph_vector_dim, input_dims),
                nn.ReLU(),
                nn.Linear(input_dims, 1),
                nn.Sigmoid()
            )
        else:
            # Do not use global sequence module
            pass

        
        """if layer_type=="bilinear":
            # use a bilinear layer to model the interaction between paragraph, and the relative positional encoding will be added at tail vector
            print("use bilinear")
            self.bilinear = nn.Bilinear(input_dims, input_dims + max_paragraph_num, 1)  # TODO: 0721 -> 0801
            self.linear = None
        
        elif layer_type=="linear":
            # use a linear layer to model the interaction between paragraph, and the relative positional encoding will be concatenated together with head vector and tail vector
            print("use linear")
            self.linear = nn.Linear(2*input_dims+max_paragraph_num, 1)
            self.bilinear = None
        
        elif layer_type=="mixed":
            # use a multihead bilinear layer to model the interaction between paragraph before a final linear projection layer, and the relative positional encoding will be added at tail vector
            print("use mixed")
            self.MEDIM = 8
            self.bilinear = nn.Bilinear(input_dims, input_dims + max_paragraph_num, self.MEDIM)
            self.activate = nn.Tanh()
            self.linear = nn.Linear(self.MEDIM, 1)

        elif layer_type=="new":
            print("use new")
            self.MEDIM = 64
            self.bilinear = nn.Bilinear(input_dims, input_dims, self.MEDIM)
            self.activate = nn.Tanh()
            self.linear = nn.Linear(self.MEDIM + max_paragraph_num + input_dims, 1)

        else:  # Multihead Attention
            # use the nn.MultiheadAttention implementation to model the interaction between paragraph
            print("use attention")
            self.mha = nn.MultiheadAttention(embed_dim=input_dims, num_heads=1,)# batch_first=True)
        self.layer_type = layer_type"""
        
        if loss_type == "margin":
            self.loss = nn.MultiMarginLoss()
        else:
            self.loss = nn.CrossEntropyLoss()
        self.loss_type = loss_type
        

    def _init_position_encoding(self, position_encoding_type, position_encoding_dim):
        """
        update self.position_encoding (nn.Module) and self.position_encoding_dim (int)
        """
        if position_encoding_dim is None:
            # for indicating the dimension of random embedding when position_encoding_dim is not provided
            position_encoding_dim = self.input_dims
        # random
        if position_encoding_type == "random":
            self.position_encoding = nn.Embedding(self.max_paragraph_num, position_encoding_dim)
            self.position_encoding_dim = position_encoding_dim
            print(f"position_encoding_type = {position_encoding_type}, position_encoding_dim = {position_encoding_dim}")

        # one-hot
        elif position_encoding_type == "one-hot":
            onehot_weight = torch.FloatTensor(torch.eye(self.max_paragraph_num))  
            self.position_encoding = nn.Embedding.from_pretrained(onehot_weight) 
            self.position_encoding_dim = self.max_paragraph_num
            print(f"position_encoding_type = {position_encoding_type}, position_encoding_dim = {self.max_paragraph_num}")

        # cosine
        elif position_encoding_type == "cosine":
            pass

        # None
        else:
            self.position_encoding = None
            self.position_encoding_dim = 0
            print(f"position_encoding_type = {None}, position_encoding_dim = {0}")
    
    def _init_position_combine_method(self, combine_method):
        if combine_method == "add":
            assert self.position_encoding_dim == self.input_dims
            def tensor_add(a, b):
                assert a.shape == b.shape
                return a + b
            self.position_encoding_combine_method = tensor_add
            self.paragraph_vector_dim = self.input_dims

        elif combine_method == "concate":
            def tensor_concate_at_last_dim(a, b):
                assert a.shape[:-1] == b.shape[:-1]
                return torch.cat((a, b), -1)
            self.position_encoding_combine_method = tensor_concate_at_last_dim
            self.paragraph_vector_dim = self.input_dims + self.position_encoding_dim

        elif combine_method == "add-linear":
            assert self.position_encoding_dim == self.input_dims
            self.position_combine_mlp = nn.Sequential(nn.Linear(self.input_dims, self.input_dims),
                                                      nn.ReLU(),
                                                      nn.Linear(self.input_dims, self.input_dims),
                                                      nn.Sigmoid()
                                                      )
            
            def tensor_add_and_linear(a, b):
                assert a.shape == b.shape
                return self.position_combine_mlp(a + b)
            self.position_encoding_combine_method = tensor_add_and_linear
            self.paragraph_vector_dim = self.input_dims

        elif combine_method == "concate-linear":
            self.position_combine_mlp = nn.Sequential(nn.Linear(self.input_dims + self.position_encoding_dim, self.input_dims + self.position_encoding_dim),
                                                      nn.ReLU(),
                                                      nn.Linear(self.input_dims + self.position_encoding_dim, self.input_dims + self.position_encoding_dim),
                                                      nn.Sigmoid()
                                                      )
            def tensor_concate_and_linear(a, b):
                assert a.shape[:-1] == b.shape[:-1]
                return self.position_combine_mlp(torch.cat((a, b), -1))
            self.position_encoding_combine_method = tensor_concate_and_linear
            self.paragraph_vector_dim = self.input_dims + self.position_encoding_dim

        else:
            self.position_encoding_combine_method = None

    def _combine_position_encoding(self, paragraph_encoding,):
        if self.position_encoding is None:
            return paragraph_encoding
        batch_size = paragraph_encoding.shape[0]
        num_paragraphs = paragraph_encoding.shape[1]
        positions = list(range(num_paragraphs))  
        positions = [positions for n in range(batch_size)]  
        positions = torch.tensor(positions, dtype=torch.long).to(paragraph_encoding.device)  
        paragraph_encoding = self.position_encoding_combine_method(paragraph_encoding, self.position_encoding(positions)) 
        return paragraph_encoding

    ##########
    def _init_relative_position_encoding(self, position_encoding_type, position_encoding_dim=None):
        """
        update self.relative_position_encoding (nn.Module) and self.relative_position_encoding_dim (int)
        """
        if position_encoding_dim is None:
            # for indicating the dimension of random embedding when position_encoding_dim is not provided
            position_encoding_dim = self.paragraph_vector_dim

        # random
        if position_encoding_type == "random":
            self.relative_position_encoding = nn.Embedding(self.max_paragraph_num, position_encoding_dim)
            self.relative_position_encoding_dim = position_encoding_dim
            print(f"relative_position_encoding_type = {position_encoding_type}, relative_position_encoding_dim = {position_encoding_dim}")

        # one-hot
        elif position_encoding_type == "one-hot":
            onehot_weight = torch.FloatTensor(torch.eye(self.max_paragraph_num))  
            self.relative_position_encoding = nn.Embedding.from_pretrained(onehot_weight)  
            self.relative_position_encoding_dim = self.max_paragraph_num
            print(f"relative_position_encoding_type = {position_encoding_type}, relative_position_encoding_dim = {self.max_paragraph_num}")

        # cosine
        elif position_encoding_type == "cosine":
            pass

        # None
        else:
            self.relative_position_encoding = None
            self.relative_position_encoding_dim = 0
            print(f"relative_position_encoding_type = {None}, relative_position_encoding_dim = {0}")
            # self.position_encoding_combine_method = None

    def _get_relative_position_encoding_for_tail(self, paragraph_encoding, tail_index):
        if self.relative_position_encoding is None:
            return torch.tensor([]).to(paragraph_encoding.device)
        batch_size = paragraph_encoding.shape[0]
        num_paragraphs = paragraph_encoding.shape[1]
        relative_positions = list(range(tail_index, 0, -1))
        relative_positions = relative_positions + [0] * (num_paragraphs - len(relative_positions))  # TODO: 0 stands for invalid father node (current node or the node after current node)
        relative_positions = [relative_positions for n in range(batch_size)]  # batchify
        relative_positions = torch.tensor(relative_positions, dtype=torch.long).to(paragraph_encoding.device)
        relative_position_encodings = self.relative_position_encoding(relative_positions)  # (num_paragraph, 200)
        # print(f"Shape of relative_position_encoding = {relative_position_encodings.shape}")
        return relative_position_encodings

    ##########
    def _init_local_pairwise_combine_method(self, combine_method, bilinear_combine="tail"):
        """
        combine_method is a `required` parameter ("None" is not allowed)
        bilinear_combine : ["head", "tail", "both"], indicating how to combine relative positional encodings with paragraph encodings in Bilinear case
        """
        if combine_method == "bilinear":
            if bilinear_combine == "head":
                print(f"local_pairwise_combine_method = Bilinear(tail_encoding, relative_position_embedding(+)head_encoding)")
                self.pairwise_bilinear_interaction = nn.Bilinear(self.paragraph_vector_dim, self.paragraph_vector_dim + self.relative_position_encoding_dim, 1)

                def concate_head_bilinear(head_encoding, tail_encoding, relative_position_encoding):
                    concated_tensor = torch.cat((head_encoding, relative_position_encoding), -1)
                    # print(f"Shape of concated_tensor = {concated_tensor.shape}")
                    return self.pairwise_bilinear_interaction(tail_encoding, concated_tensor)
                self.local_pairwise_combine_method = concate_head_bilinear
            
            elif bilinear_combine == "tail":
                print(f"local_pairwise_combine_method = Bilinear(head_encoding, relative_position_embedding(+)tail_encoding)")
                self.pairwise_bilinear_interaction = nn.Bilinear(self.paragraph_vector_dim, self.paragraph_vector_dim + self.relative_position_encoding_dim, 1)

                def concate_tail_bilinear(head_encoding, tail_encoding, relative_position_encoding):
                    concated_tensor = torch.cat((tail_encoding, relative_position_encoding), -1)
                    # print(f"Shape of concated_tensor = {concated_tensor.shape}")
                    return self.pairwise_bilinear_interaction(head_encoding, concated_tensor)
                self.local_pairwise_combine_method = concate_tail_bilinear
            
            else:  # both
                print(f"local_pairwise_combine_method = Bilinear(relative_position_embedding(+)head_encoding, relative_position_embedding(+)tail_encoding)")
                self.pairwise_bilinear_interaction = nn.Bilinear(self.paragraph_vector_dim + self.relative_position_encoding_dim, self.paragraph_vector_dim + self.relative_position_encoding_dim, 1)

                def concate_both_bilinear(head_encoding, tail_encoding, relative_position_encoding):
                    concated_head_tensor = torch.cat((head_encoding, relative_position_encoding), -1)
                    concated_tail_tensor = torch.cat((tail_encoding, relative_position_encoding), -1)
                    # print(f"Shape of concated_head_tensor = {concated_head_tensor.shape}")
                    # print(f"Shape of concated_tail_tensor = {concated_tail_tensor.shape}")
                    return self.pairwise_bilinear_interaction(concated_head_tensor, concated_tail_tensor)
                self.local_pairwise_combine_method = concate_both_bilinear
            # self.local_pairwise_combine_method = 

        elif combine_method == "concate-linear":
            print(f"local_pairwise_combine_method = Linear(head_encoding(+)relative_position_embedding(+)tail_encoding)")
            self.pairwise_linear_interaction = nn.Linear(2*self.paragraph_vector_dim + self.relative_position_encoding_dim, 1)

            def concate_linear(head_encoding, tail_encoding, relative_position_encoding):
                concated_tensor = torch.cat((head_encoding, tail_encoding, relative_position_encoding), -1)
                return self.pairwise_linear_interaction(concated_tensor)
            self.local_pairwise_combine_method = concate_linear

        elif combine_method == "concate-mlp":
            print(f"local_pairwise_combine_method = MLP(head_encoding(+)relative_position_embedding(+)tail_encoding)")
            self.pairwise_mlp_interaction = nn.Sequential(nn.Linear(2*self.paragraph_vector_dim + self.relative_position_encoding_dim, self.input_dims),
                                                          nn.ReLU(),
                                                          nn.Linear(self.input_dims, 1),
                                                        #   nn.Sigmoid()
                                                          )

            def concate_mlp(head_encoding, tail_encoding, relative_position_encoding):
                concated_tensor = torch.cat((head_encoding, tail_encoding, relative_position_encoding), -1)
                return self.pairwise_mlp_interaction(concated_tensor)
            self.local_pairwise_combine_method = concate_mlp

        elif combine_method == "trilinear":
            # self.pairwise_trilinear_interaction = 
            pass


    def _combine_local_pairwise_information(self, head_vectors, tail_vectors, tail_index,):
        batch_size = tail_vectors.shape[0]
        num_paragraphs = tail_vectors.shape[1]
        tails = tail_vectors[:,tail_index,:].repeat(1, num_paragraphs, 1)  # (Batch, num_paragraph, hidden_dim)
        relative_position_encoding = self._get_relative_position_encoding_for_tail(tail_vectors, tail_index=tail_index)
        # print(f"Shape of head_vectors = {head_vectors.shape}")
        # print(f"Shape of tails = {tails.shape}")
        # print(f"Shape of relative_position_encoding = {relative_position_encoding.shape}")
        father_node_logit_vector = self.local_pairwise_combine_method(head_vectors, tails, relative_position_encoding).permute(0,2,1)  # (Batch, num_paragraph, 1) -> (Batch, 1, num_paragraph)
        # print(f"Shape of father_node_logit_vector = {father_node_logit_vector.shape}")
        return father_node_logit_vector
    
    ##########
    def _init_lstm(self, lstm_input_type, ):
        pass
    def _init_lstm_combine_method(self, combine_method):
        pass
    def _get_lstm_initial_state(self):
        pass
    def _combine_global_information(self):
        pass
    
    def forward(self, paragraphs, golden=None):
        """
        `remember you still need to take batch dimension into consideration even if batch size = 1`
        Doing `masked` self attention
        :param paragraphs: (batch_size, num_paragraphs, hidden_dim)
        :param padding: upper triangular matrix, (batch_size, num_paragraphs, num_paragraphs)
        :param golden(Optional): (batch_size, num_paragraphs, num_paragraphs)  ### (batch_size, num_paragraphs)
        :return: father_node_scores: (batch_size, num_paragraphs, num_paragraphs)
        :return: loss: scalar
        """
        if len(paragraphs.shape) == 2:
            paragraphs = paragraphs.unsqueeze(dim=0)
        assert len(paragraphs.shape) == 3
        batch_size, num_paragraphs, hidden_dim = paragraphs.shape

        loss = None
        head_vectors = self.head_mlp(paragraphs)  # (Batch, num_paragraph, hidden_dim)
        tail_vectors = self.tail_mlp(paragraphs)  # (Batch, num_paragraph, hidden_dim)
        
        head_vectors = self._combine_position_encoding(head_vectors)
        tail_vectors = self._combine_position_encoding(tail_vectors)


        ## >>> >>> ##
        father_node_logit_matrix = torch.zeros(batch_size, num_paragraphs, num_paragraphs).to(paragraphs.device)
        # lstm_hidden = torch.zeros(self.max_paragraph_num).to(paragraphs.device) 
        # lstm_cell = torch.zeros(self.max_paragraph_num).to(paragraphs.device) 
        if self.use_global_sequence_module:
            lstm_hidden = torch.zeros(self.paragraph_vector_dim).to(paragraphs.device) 
            lstm_cell = torch.zeros(self.paragraph_vector_dim).to(paragraphs.device)
        for t in range(num_paragraphs):
            father_node_logit_matrix[:,t,:] = self._combine_local_pairwise_information(head_vectors, tail_vectors, tail_index=t)
                
            if self.use_global_sequence_module:
                print("use_global_sequence_module")
                interact_tensor = torch.cat((head_vectors, lstm_hidden.repeat(1, num_paragraphs, 1)), dim=2) 
                # print(f"Shape of interact_tensor = {interact_tensor.shape}")
                probability_scores = self.score_mlp(interact_tensor)
                probability_scores = probability_scores.permute(0,2,1)  # (Batch, num_paragraph, 1) -> (Batch, 1, num_paragraph)
                # print(f"Shape of probability_scores = {probability_scores.shape}")

                father_node_logit_matrix[:,t,:] = father_node_logit_matrix[:,t,:].clone() * probability_scores 
                father_node_predict = torch.argmax(father_node_logit_matrix[:,t,:t]) if t > 0 else 0 
                # lstm_cell, lstm_hidden = self.iterlstm(head_vectors[0,father_node_predict,:], lstm_cell, lstm_hidden) 
                # lstm_cell, lstm_hidden = self.iterlstm(self.position_encoding(torch.tensor(father_node_predict).to(paragraphs.device)), lstm_cell, lstm_hidden)
                lstm_cell, lstm_hidden = self.iterlstm(head_vectors[:,father_node_predict,:], lstm_cell, lstm_hidden)
                # print(f"Shape of lstm_cell = {lstm_cell.shape}")
                # print(f"Shape of lstm_hidden = {lstm_hidden.shape}")
        # attn_mask = torch.triu(torch.ones(num_paragraphs, num_paragraphs)).to(paragraphs.device)
        attn_mask = (torch.ones(num_paragraphs, num_paragraphs) - torch.tril(torch.ones(num_paragraphs, num_paragraphs))).to(paragraphs.device)  # here, num_paragraphs is num edus + dummy node
        # father_node_logit_matrix = father_node_logit_matrix + (1 - attn_mask + (-10e9) * attn_mask)  # 1106  # 1121
        father_node_logit_matrix = father_node_logit_matrix + ((-10e9) * attn_mask)
        ## >>> >>> ##


        father_node_logit_scores = torch.softmax(father_node_logit_matrix[:, 1:, :], dim=2)  # wipe off dummy node, but keep the sum of probabilities = 1
        # print(father_node_logit_scores.shape)

        # nn.MarginRankingLoss
        # print(golden)
        if golden is not None:
            # Compute loss
            flatten_golden = torch.flatten(golden)
            flatten_golden = flatten_golden + 1  # Deal with "NA" type
            flatten_father_node_logit_scores = torch.flatten(father_node_logit_scores, end_dim=1)
            flatten_father_node_logit_matrix = torch.flatten(father_node_logit_matrix[:, 1:, :], end_dim=1)  # TODO
            # print(flatten_father_node_logit_scores.shape)
            # print(flatten_golden)
            # print(flatten_golden.shape)
            # print(flatten_father_node_logit_matrix.shape)
            # print(flatten_father_node_logit_scores)
            # loss = self.loss(flatten_father_node_logit_scores, flatten_golden)
            if self.loss_type == "ce":
                loss = self.loss(flatten_father_node_logit_matrix, flatten_golden)
                # print(flatten_father_node_logit_matrix)
                # print(flatten_golden)
                # print(loss)
            else:
                loss = self.loss(flatten_father_node_logit_scores, flatten_golden)

        outputs = (father_node_logit_scores, loss)
        return outputs


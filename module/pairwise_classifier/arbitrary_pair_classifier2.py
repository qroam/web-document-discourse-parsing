# -*- coding: utf-8 -*-
from typing import List
import torch
from torch import nn
from collections import defaultdict

from .biaffine import BiaffineAttention, BiaffinePointerAttention



class ArbitraryPairClassifier(nn.Module):
    """
    Predicting the Continuity ["Continue", "Break", "Combine"(Optional)] between same-level adjancent paragraphs
    Mechanism: Doing Concat(head, tail) -> Linear() to projection the into label space
    """
    # def __init__(self, input_dims, output_dims, position_highlight_mlp=False):
    def __init__(self, head_input_dims, tail_input_dims, output_dims, pair_interation_method="bilinear", position_highlight_mlp=False, relative_position_embedding=None):
        super().__init__()
        # self.input_dims = input_dims
        self.head_input_dims = head_input_dims
        self.tail_input_dims = tail_input_dims
        self.output_dims = output_dims
        
        self.position_highlight_mlp = position_highlight_mlp
        self.head_mlp = nn.Sequential(nn.Linear(head_input_dims, head_input_dims),
                                      nn.ReLU(),
                                      nn.Linear(head_input_dims, head_input_dims),
                                      nn.Tanh()
                                      ) if position_highlight_mlp else lambda x: x  # 11/30
        self.tail_mlp = nn.Sequential(nn.Linear(tail_input_dims, tail_input_dims),
                                      nn.ReLU(),
                                      nn.Linear(tail_input_dims, tail_input_dims),
                                      nn.Tanh()
                                      ) if position_highlight_mlp else lambda x: x  # 11/30
        
        # self.mlp = nn.Sequential(nn.Linear(concate_input_dims, concate_input_dims),
        #                          nn.Tanh(),
        #                          nn.Linear(concate_input_dims, output_dims),
        #                          )
        

        # self.relative_position_encoding = nn.Embedding(self.max_paragraph_num, position_encoding_dim)
        self.relative_position_encoding = relative_position_embedding
        self.relative_position_encoding_dims = relative_position_embedding.embedding_dim if relative_position_embedding else 0

        self.pair_interation_method = pair_interation_method
        self._init_pairwise_interaction_method(method=pair_interation_method, bilinear_combine="tail")

        self.loss = nn.CrossEntropyLoss()

        print("initialized ArbitraryPairClassifier")

    def _concate(self, head_paragraphs, tail_paragraphs, previous_ids=None):
        # Depracated
        """
        :param head_paragraphs: (batch, max_node_num, hidden_dim_for_head_representations) or (batch, hidden_dim_for_head_representations)
        :param tail_paragraphs: (batch, max_node_num, hidden_dim_for_tail_representations) or (batch, hidden_dim_for_tail_representations)
        :param previous_ids: (batch, max_node_num), indicating for each tail paragraph its corresponding head paragraph index, 0 means dummy node and padding!!!
        if want to predict each pair incrementally, you can just input max_node_num=1 and previous_ids=None
        """

        if len(head_paragraphs.shape) == 2:  # TODO
            assert len(tail_paragraphs.shape) == 2
            batch_size, head_hidden_dim = head_paragraphs.shape
            batch_size, tail_hidden_dim = tail_paragraphs.shape
        elif len(head_paragraphs.shape) == 3:
            assert len(tail_paragraphs.shape) == 3
            batch_size, max_node_num, head_hidden_dim = head_paragraphs.shape
            batch_size, max_node_num, tail_hidden_dim = tail_paragraphs.shape
        # TODO: we want the concatenation progress to be parallel, i.e. do not use for loop as possible
        ##mask = (previous_ids == 0).long()
        # print(previous_ids.shape)  # [1, 30]
        # print(previous_ids)
        # previous_paragraphs = paragraphs[:][previous_ids][:]
        if previous_ids is None:
            previous_ids = torch.tensor([list(range(max_node_num))], dtype=torch.int).to(head_paragraphs.device)  # TODO: batchify implementation
        print("head_paragraphs_vector", head_paragraphs[:,:,:5])
        previous_paragraphs = head_paragraphs[:, previous_ids.squeeze(0), :]
        print("previous_paragraphs_vector", previous_paragraphs[:,:,:5])
        # print(previous_ids)
        # print(previous_paragraphs.shape)  # [1, 30, 30, 768]
        # print(tail_paragraphs.shape)  # [1, 30, 768]
        concated_paragraphs = torch.cat((previous_paragraphs, tail_paragraphs), dim=2)  # caution for the dim
        return concated_paragraphs##, mask
    
    def _sequence_align(self, head_paragraphs, previous_ids=None):
        """
        :param head_paragraphs: (batch, max_node_num, hidden_dim_for_head_representations) or (batch, hidden_dim_for_head_representations)
        :param tail_paragraphs: (batch, max_node_num, hidden_dim_for_tail_representations) or (batch, hidden_dim_for_tail_representations)
        :param previous_ids: (batch, max_node_num), indicating for each tail paragraph its corresponding head paragraph index, 0 means dummy node and padding!!!
        if want to predict each pair incrementally, you can just input max_node_num=1 and previous_ids=None
        """

        if len(head_paragraphs.shape) == 2:  # TODO
            # assert len(tail_paragraphs.shape) == 2
            batch_size, head_hidden_dim = head_paragraphs.shape
            # batch_size, tail_hidden_dim = tail_paragraphs.shape
        elif len(head_paragraphs.shape) == 3:
            # assert len(tail_paragraphs.shape) == 3
            batch_size, max_node_num, head_hidden_dim = head_paragraphs.shape
            # batch_size, max_node_num, tail_hidden_dim = tail_paragraphs.shape
        # TODO: we want the concatenation progress to be parallel, i.e. do not use for loop as possible
        ##mask = (previous_ids == 0).long()
        # print(previous_ids.shape)  # [1, 30]
        # print(previous_ids)
        # previous_paragraphs = paragraphs[:][previous_ids][:]
        if previous_ids is None:
            return head_paragraphs  # 12/2
            previous_ids = torch.tensor([list(range(max_node_num))], dtype=torch.int).to(head_paragraphs.device)  # TODO: batchify implementation
        # print("head_paragraphs_vector", head_paragraphs[:,:,:5])
        previous_paragraphs = head_paragraphs[:, previous_ids.squeeze(0), :]
        # print("previous_paragraphs_vector", previous_paragraphs[:,:,:5])
        # print(previous_ids)
        # print(previous_paragraphs.shape)  # [1, 30, 30, 768]
        # print(tail_paragraphs.shape)  # [1, 30, 768]
        return previous_paragraphs

    

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
    
    def _get_relative_position_encoding(self, previous_ids):
        """
        : params previous_ids: (batch_size, num_paragraphs), 0 is for dummy head node or padding tail node
        """
        # relative_positions = list(range(tail_index, 0, -1))
        # relative_positions = relative_positions + [0] * (num_paragraphs - len(relative_positions))  # TODO: 0 stands for invalid father node (current node or the node after current node)
        # relative_positions = [relative_positions for n in range(batch_size)]  # batchify
        # relative_positions = torch.tensor(relative_positions, dtype=torch.long).to(paragraph_encoding.device)
        batch_size, num_paragraphs = previous_ids.shape
        tail_ids = torch.arange(num_paragraphs).expand(batch_size, num_paragraphs).to(previous_ids.device)
        relative_positions = tail_ids - previous_ids
        relative_position_encodings = self.relative_position_encoding(relative_positions)  # (num_paragraph, 200)
        # print(f"Shape of relative_position_encoding = {relative_position_encodings.shape}")
        return relative_position_encodings

    ##########
    # def _init_local_pairwise_combine_method(self, combine_method, bilinear_combine="tail"):
    def _init_pairwise_interaction_method(self, method, bilinear_combine="tail"):
        """
        combine_method is a `required` parameter ("None" is not allowed)
        bilinear_combine : ["head", "tail", "both"], indicating how to combine relative positional encodings with paragraph encodings in Bilinear case
        """
        namecards = {
            "bilinear_head": "pairwise_interaction_method = Bilinear(tail_encoding, relative_position_embedding(+)head_encoding)",
            "bilinear_tail": "pairwise_interaction_method = Bilinear(head_encoding, relative_position_embedding(+)tail_encoding)",
            "bilinear_both": f"pairwise_interaction_method = Bilinear(relative_position_embedding(+)head_encoding, relative_position_embedding(+)tail_encoding)",
            # "biaffine_head": "pairwise_interaction_method = BiAffine(tail_encoding, relative_position_embedding(+)head_encoding)"
            # "biaffine_tail": "pairwise_interaction_method = BiAffine(head_encoding, relative_position_embedding(+)tail_encoding)",
            # "biaffine_both": f"pairwise_interaction_method = BiAffine(relative_position_embedding(+)head_encoding, relative_position_embedding(+)tail_encoding)",
            "biaffine": "pairwise_interaction_method = BiAffine(head_encoding, relative_position_embedding(+)tail_encoding)",
            "concate-linear": f"pairwise_interaction_method = Linear(head_encoding(+)relative_position_embedding(+)tail_encoding)",
            "concate-mlp": f"pairwise_interaction_method = MLP(head_encoding(+)relative_position_embedding(+)tail_encoding)",
        }
        print(namecards[method+"_"+bilinear_combine] if method in ["bilinear",] else namecards[method])

        if method == "bilinear":
            if bilinear_combine == "head":
                self.bilinear = nn.Bilinear(self.tail_input_dims, self.head_input_dims + self.relative_position_encoding_dims, self.output_dims)

                def concate_head_bilinear(head_paragraphs, tail_paragraphs, relative_position_encoding):
                    concated_tensor = torch.cat((head_paragraphs, relative_position_encoding), -1)
                    # print(f"Shape of concated_tensor = {concated_tensor.shape}")
                    return self.bilinear(tail_paragraphs, concated_tensor)
                self._local_pairwise_combine_method = concate_head_bilinear
            
            elif bilinear_combine == "tail":
                self.bilinear = nn.Bilinear(self.head_input_dims, self.tail_input_dims + self.relative_position_encoding_dims, self.output_dims)

                def concate_tail_bilinear(head_paragraphs, tail_paragraphs, relative_position_encoding):
                    concated_tensor = torch.cat((tail_paragraphs, relative_position_encoding), -1)
                    # print(f"Shape of concated_tensor = {concated_tensor.shape}")
                    return self.bilinear(head_paragraphs, concated_tensor)
                self._local_pairwise_combine_method = concate_tail_bilinear
            
            else:  # both
                self.bilinear = nn.Bilinear(self.head_input_dims + self.relative_position_encoding_dims, self.tail_input_dims + self.relative_position_encoding_dims, 1)

                def concate_both_bilinear(head_paragraphs, tail_paragraphs, relative_position_encoding):
                    concated_head_tensor = torch.cat((head_paragraphs, relative_position_encoding), -1)
                    concated_tail_tensor = torch.cat((tail_paragraphs, relative_position_encoding), -1)
                    # print(f"Shape of concated_head_tensor = {concated_head_tensor.shape}")
                    # print(f"Shape of concated_tail_tensor = {concated_tail_tensor.shape}")
                    return self.bilinear(concated_head_tensor, concated_tail_tensor)
                self._local_pairwise_combine_method = concate_both_bilinear
            # self.local_pairwise_combine_method = 

        elif method == "biaffine":
            self.biaffine = BiaffineAttention(self.head_input_dims, self.tail_input_dims + self.relative_position_encoding_dims, self.output_dims)
            def concate_tail_biaffine(head_paragraphs, tail_paragraphs, relative_position_encoding):
                concated_tensor = torch.cat((tail_paragraphs, relative_position_encoding), -1)
                return self.biaffine(head_paragraphs, concated_tensor)
            self._local_pairwise_combine_method = concate_tail_biaffine
        
        elif method == "concate-linear":
            self.linear = nn.Linear(self.head_input_dims + self.tail_input_dims + self.relative_position_encoding_dims, self.output_dims)
            # self.pairwise_linear_interaction = nn.Linear(2*self.paragraph_vector_dim + self.relative_position_encoding_dim, 1)

            def concate_linear(head_paragraphs, tail_paragraphs, relative_position_encoding):
                concated_paragraphs = torch.cat((head_paragraphs, tail_paragraphs, relative_position_encoding), dim=2)
                # concated_paragraphs = self._concate(head_paragraphs, tail_paragraphs, previous_ids)  # (batch_size, num_paragraphs, head_hidden_dim+tail_hidden_dim)
                logits = self.mlp(concated_paragraphs)
                # concated_tensor = torch.cat((head_encoding, tail_encoding, relative_position_encoding), -1)
                # return self.pairwise_linear_interaction(concated_tensor)
                return logits
            self._local_pairwise_combine_method = concate_linear

        elif method == "concate-mlp":
            concate_input_dims = self.head_input_dims + self.tail_input_dims + self.relative_position_encoding_dims
            self.mlp = nn.Sequential(nn.Linear(concate_input_dims, concate_input_dims),
                                     nn.Tanh(),
                                     nn.Linear(concate_input_dims, self.output_dims),
                                     )
            # self.pairwise_mlp_interaction = nn.Sequential(nn.Linear(2*self.paragraph_vector_dim + self.relative_position_encoding_dim, self.input_dims),
            #                                               nn.ReLU(),
            #                                               nn.Linear(self.input_dims, 1),
            #                                               nn.Sigmoid()
            #                                               )

            def concate_mlp(head_paragraphs, tail_paragraphs, relative_position_encoding):
                concated_paragraphs = torch.cat((head_paragraphs, tail_paragraphs, relative_position_encoding), dim=2)
                # concated_paragraphs = self._concate(head_paragraphs, tail_paragraphs, previous_ids)  # (batch_size, num_paragraphs, head_hidden_dim+tail_hidden_dim)
                logits = self.mlp(concated_paragraphs)  # (batch_size, num_paragraphs, output_dims)
                # concated_tensor = torch.cat((head_encoding, tail_encoding, relative_position_encoding), -1)
                # return self.pairwise_mlp_interaction(concated_tensor)
                return logits
            self._local_pairwise_combine_method = concate_mlp

        # elif combine_method == "trilinear":
        #     # self.pairwise_trilinear_interaction = 
        #     pass


    # def _combine_local_pairwise_information(self, head_paragraphs, tail_paragraphs, previous_ids=None,):
    def _pairwise_interaction(self, head_paragraphs, tail_paragraphs, previous_ids=None, relative_positions=None):
        batch_size, num_paragraphs = tail_paragraphs.shape[:2]
        # tails = tail_vectors[:,tail_index,:].repeat(1, num_paragraphs, 1)  # (Batch, num_paragraph, hidden_dim)
        # print(tail_paragraphs.shape)
        # print(relative_positions)
        if relative_positions is not None:
            relative_position_encoding = self.relative_position_encoding(relative_positions) if self.relative_position_encoding else torch.tensor([[[]]*num_paragraphs]*batch_size).to(head_paragraphs.device)

        else:
            relative_position_encoding = self._get_relative_position_encoding(previous_ids=previous_ids) if self.relative_position_encoding else torch.tensor([[[]]*num_paragraphs]*batch_size).to(head_paragraphs.device)
        # print(f"Shape of head_vectors = {head_vectors.shape}")
        # print(f"Shape of tails = {tails.shape}")
        # print(f"Shape of relative_position_encoding = {relative_position_encoding.shape}")
        logits = self._local_pairwise_combine_method(head_paragraphs, tail_paragraphs, relative_position_encoding)  # (Batch, num_paragraph, 1) -> (Batch, 1, num_paragraph)
        # print(f"Shape of father_node_logit_vector = {father_node_logit_vector.shape}")
        return logits
    
    
    def forward_without_dummy_node(self, head_paragraphs, tail_paragraphs, previous_ids=None, golden=None, relative_positions=None):
        """
        Doing masked self attention
        :param paragraphs: (batch_size, num_paragraphs, hidden_dim)
        :param previous_ids: (batch_size, num_paragraphs), 0 is for dummy head node or padding tail node
        :param golden: (batch_size, num_paragraphs)
        :return: previous_node_logits: (batch_size, num_paragraphs, output_dims)
        """
        loss = None


        head_paragraphs = self.head_mlp(head_paragraphs)  # (Batch, num_paragraph, hidden_dim) # 11/30
        tail_paragraphs = self.tail_mlp(tail_paragraphs)  # (Batch, num_paragraph, hidden_dim) # 11/30


        """concated_paragraphs = self._concate(head_paragraphs, tail_paragraphs, previous_ids)  # (batch_size, num_paragraphs, head_hidden_dim+tail_hidden_dim)
        logits = self.mlp(concated_paragraphs)  # (batch_size, num_paragraphs, output_dims)"""
        head_paragraphs_aligned = self._sequence_align(head_paragraphs, previous_ids)


        logits = self._pairwise_interaction(head_paragraphs_aligned, tail_paragraphs, previous_ids, relative_positions)
        
        if previous_ids is not None:
            previous_mask = previous_ids.clone()
            previous_mask[previous_mask != 0] = 1
            previous_mask = previous_mask.detach()
        else:
            previous_mask = torch.ones(head_paragraphs.shape[0], head_paragraphs.shape[1]).to(dtype=torch.bool, device=head_paragraphs.device)

        previous_relation_scores = torch.softmax(logits, dim=2)
        
        if golden is not None:
            """golden = golden[previous_mask != 0]
            logits = logits[previous_mask != 0]
            loss = self.loss(logits, golden)"""
            masked_logits = logits[golden != -1] 
            masked_golden = golden[golden != -1]
            if masked_logits.shape[0] != 0:
                loss = self.loss(masked_logits, masked_golden)
            else:
                loss = torch.tensor(0).to(masked_logits.device)
        outputs = (previous_relation_scores, loss)
        return outputs
    

    def forward_with_graph_encodings(self, directed_graph_encodings, previous_ids, golden=None):
        """
        Doing masked self attention
        :param directed_graph_encodings_head_to_tail: (batch, max_node_num+1, max_node_num+1, self.input_dims=params.path_hidden_size)
        :param directed_graph_encodings_tail_to_head: (batch, max_node_num+1, max_node_num+1, self.input_dims=params.path_hidden_size)
        :param previous_ids: (batch_size, max_node_num+1, 1),
        :param golden: (batch_size, max_node_num)
        :return: previous_node_logits: (batch_size, num_paragraphs, 3)
        """
        loss = None
        # print(directed_graph_encodings.shape)
        # print(previous_ids.shape)


        # concated_paragraphs, mask = self.concate(paragraphs, previous_ids)  # (batch_size, num_paragraphs, 2 * hidden_dim)
        # logits = self.mlp(concated_paragraphs)  # (batch_size, num_paragraphs, 3)

        # pair_node_encodings = directed_graph_encodings[previous_ids]
        # pair_node_encodings = directed_graph_encodings[:, :, previous_ids, :]
        batch_size, max_node_num, max_node_num, hidden_dim = directed_graph_encodings.shape
        pair_node_encodings_tail_to_head = torch.zeros(batch_size, max_node_num, hidden_dim).to(directed_graph_encodings.device)
        pair_node_encodings_head_to_tail = torch.zeros(batch_size, max_node_num, hidden_dim).to(directed_graph_encodings.device)
        relative_positions = torch.zeros(batch_size, max_node_num).to(directed_graph_encodings.device)

        for i in range(previous_ids.shape[0]):
            for j in range(previous_ids.shape[1]):  # j: tail node id
                pair_node_encodings_tail_to_head[i, j] = directed_graph_encodings[i, j, previous_ids[i][j]]  # previous_ids[i][j]: head node id
                pair_node_encodings_head_to_tail[i, j] = directed_graph_encodings[i, previous_ids[i][j], j]
                relative_positions[i, j] = j - previous_ids[i][j]
        # directed_graph_encodings[:, :, previous_ids, :]
        # print(pair_node_encodings.shape)
        # print(relative_positions)
        return self.forward(
            pair_node_encodings_head_to_tail,
            pair_node_encodings_tail_to_head,
            previous_ids=previous_ids,  # 1/2, None
            golden=golden,
            relative_positions=relative_positions
        )
    
    def forward_with_graph_encodings_return_logits(self, directed_graph_encodings,):
        # 12/31 This is called by forward() of SSA
        """
        Doing masked self attention
        :param directed_graph_encodings: (batch, max_node_num+1, max_node_num+1, self.input_dims=params.path_hidden_size)
        :param previous_ids: (batch_size, max_node_num+1, 1),
        :param golden: (batch_size, max_node_num)
        :return: previous_node_logits: (batch_size, num_paragraphs, 3)
        """
        loss = None

        batch_size, num_paragraphs = directed_graph_encodings.shape[:2]
        path_hidden_size = directed_graph_encodings.shape[-1]
        
        directed_graph_encodings_head_to_tail = directed_graph_encodings
        directed_graph_encodings_tail_to_head = directed_graph_encodings.transpose(1, 2)
        directed_graph_encodings_head_to_tail = self.head_mlp(directed_graph_encodings_head_to_tail)
        directed_graph_encodings_tail_to_head = self.tail_mlp(directed_graph_encodings_tail_to_head)

        """concated_paragraphs = self._concate(head_paragraphs, tail_paragraphs, previous_ids)  # (batch_size, num_paragraphs, head_hidden_dim+tail_hidden_dim)
        logits = self.mlp(concated_paragraphs)  # (batch_size, num_paragraphs, output_dims)"""
        # head_paragraphs_aligned = self._sequence_align(head_paragraphs, previous_ids)
        # logits = self._pairwise_interaction(head_paragraphs_aligned, tail_paragraphs, previous_ids)

        ## pointer interaction ##
        # logits = self._pairwise_interaction(head_paragraphs_aligned, tail_paragraphs, previous_ids, relative_positions)
        # father_relation_logit_matrix = torch.zeros(batch_size, num_paragraphs, num_paragraphs).to(directed_graph_encodings.device)  # 11/30 self.output_dims

        # relative_positions = torch.zeros(batch_size, max_node_num).to(directed_graph_encodings.device)
        # relative_positions = torch.arangezeros(batch_size, max_node_num).to(directed_graph_encodings.device)  # TODO
        
        father_relation_logit_matrix = self._pairwise_interaction(  # 12/31
            directed_graph_encodings_head_to_tail.reshape(batch_size, -1, path_hidden_size),
            directed_graph_encodings_tail_to_head.reshape(batch_size, -1, path_hidden_size),
            previous_ids=None,
            relative_positions=None,
        ).reshape(batch_size, num_paragraphs, num_paragraphs, -1)

                
        return father_relation_logit_matrix

    
    def forward(self, head_paragraphs, tail_paragraphs, previous_ids=None, golden=None, relative_positions=None):
        """
        Doing masked self attention
        :param paragraphs: (batch_size, num_paragraphs, hidden_dim)
        :param previous_ids: (batch_size, num_paragraphs), 0 is for dummy head node or padding tail node
        :param golden: (batch_size, num_paragraphs)
        :return: previous_node_logits: (batch_size, num_paragraphs, output_dims)
        """
        loss = None


        head_paragraphs = self.head_mlp(head_paragraphs)  # (Batch, num_paragraph, hidden_dim) # 11/30
        tail_paragraphs = self.tail_mlp(tail_paragraphs)  # (Batch, num_paragraph, hidden_dim) # 11/30


        """concated_paragraphs = self._concate(head_paragraphs, tail_paragraphs, previous_ids)  # (batch_size, num_paragraphs, head_hidden_dim+tail_hidden_dim)
        logits = self.mlp(concated_paragraphs)  # (batch_size, num_paragraphs, output_dims)"""
        head_paragraphs_aligned = self._sequence_align(head_paragraphs, previous_ids)

        # print(head_paragraphs_aligned)


        logits = self._pairwise_interaction(head_paragraphs_aligned, tail_paragraphs, previous_ids, relative_positions)
        # print(logits)
        # print(previous_ids)
        # print(previous_ids.shape)
        # if previous_ids is None:
        # print("previous_ids:", previous_ids)
        # print("golden:", golden)
        if previous_ids is not None:
            previous_mask = previous_ids.clone()
            previous_mask[previous_mask != 0] = 1
            previous_mask = previous_mask.detach()
        else:
            previous_mask = torch.ones(head_paragraphs.shape[0], head_paragraphs.shape[1]).to(dtype=torch.bool, device=head_paragraphs.device)

        previous_mask = previous_mask[:,1:]  # wipe off dummy node
        logits = logits[:,1:,:]  # wipe off dummy node
        # print("previous_mask:", previous_mask)
        # print("logits:", logits)
        previous_relation_scores = torch.softmax(logits, dim=2)
        # print(golden.shape)
        # print(logits.shape)
        # print(previous_mask.shape)

        if golden is not None:
            # golden = golden * previous_mask
            # TODO
            # logits[:][previous_mask == 0][:] = [1., 0., 0., 0.]  # Mask out the nodes which have no previous node in tree
            # print(golden.shape)
            # print(logits.shape)
            # print(previous_mask.shape)
            golden = golden[previous_mask != 0]
            logits = logits[previous_mask != 0]
            loss = self.loss(logits, golden)

        outputs = (previous_relation_scores, loss)
        return outputs



class ArbitraryPairPointer(nn.Module):
    """
    Predicting the Continuity ["Continue", "Break", "Combine"(Optional)] between same-level adjancent paragraphs
    Mechanism: Doing Concat(head, tail) -> Linear() to projection the into label space
    """
    # def __init__(self, input_dims, output_dims, position_highlight_mlp=False):
    def __init__(self, head_input_dims, tail_input_dims, output_dims, pair_interation_method="bilinear", position_highlight_mlp=False, relative_position_embedding=None, loss_type="ce"):
        super().__init__()
        # self.input_dims = input_dims
        self.head_input_dims = head_input_dims
        self.tail_input_dims = tail_input_dims
        self.output_dims = output_dims
        
        self.position_highlight_mlp = position_highlight_mlp
        self.head_mlp = nn.Sequential(nn.Linear(head_input_dims, head_input_dims),
                                      nn.ReLU(),
                                      nn.Linear(head_input_dims, head_input_dims),
                                      nn.Tanh()
                                      ) if position_highlight_mlp else lambda x: x  # 11/30
        self.tail_mlp = nn.Sequential(nn.Linear(tail_input_dims, tail_input_dims),
                                      nn.ReLU(),
                                      nn.Linear(tail_input_dims, tail_input_dims),
                                      nn.Tanh()
                                      ) if position_highlight_mlp else lambda x: x  # 11/30
        
        # self.mlp = nn.Sequential(nn.Linear(concate_input_dims, concate_input_dims),
        #                          nn.Tanh(),
        #                          nn.Linear(concate_input_dims, output_dims),
        #                          )
        

        # self.relative_position_encoding = nn.Embedding(self.max_paragraph_num, position_encoding_dim)
        self.relative_position_encoding = relative_position_embedding
        self.relative_position_encoding_dims = relative_position_embedding.embedding_dim if relative_position_embedding else 0

        self.pair_interation_method = pair_interation_method
        self._init_pairwise_interaction_method(method=pair_interation_method, bilinear_combine="tail")

        if loss_type == "margin":
            self.loss = nn.MultiMarginLoss()
        else:
            self.loss = nn.CrossEntropyLoss()
        self.loss_type = loss_type

        print("initialized ArbitraryPairPointer")

    def _concate(self, head_paragraphs, tail_paragraphs, previous_ids=None):
        # Depracated
        """
        :param head_paragraphs: (batch, max_node_num, hidden_dim_for_head_representations) or (batch, hidden_dim_for_head_representations)
        :param tail_paragraphs: (batch, max_node_num, hidden_dim_for_tail_representations) or (batch, hidden_dim_for_tail_representations)
        :param previous_ids: (batch, max_node_num), indicating for each tail paragraph its corresponding head paragraph index, 0 means dummy node and padding!!!
        if want to predict each pair incrementally, you can just input max_node_num=1 and previous_ids=None
        """

        if len(head_paragraphs.shape) == 2:  # TODO
            assert len(tail_paragraphs.shape) == 2
            batch_size, head_hidden_dim = head_paragraphs.shape
            batch_size, tail_hidden_dim = tail_paragraphs.shape
        elif len(head_paragraphs.shape) == 3:
            assert len(tail_paragraphs.shape) == 3
            batch_size, max_node_num, head_hidden_dim = head_paragraphs.shape
            batch_size, max_node_num, tail_hidden_dim = tail_paragraphs.shape
        # TODO: we want the concatenation progress to be parallel, i.e. do not use for loop as possible
        ##mask = (previous_ids == 0).long()
        # print(previous_ids.shape)  # [1, 30]
        # print(previous_ids)
        # previous_paragraphs = paragraphs[:][previous_ids][:]
        if previous_ids is None:
            previous_ids = torch.tensor([list(range(max_node_num))], dtype=torch.int).to(head_paragraphs.device)  # TODO: batchify implementation
        # print("head_paragraphs_vector", head_paragraphs[:,:,:5])
        previous_paragraphs = head_paragraphs[:, previous_ids.squeeze(0), :]
        # print("previous_paragraphs_vector", previous_paragraphs[:,:,:5])
        # print(previous_ids)
        # print(previous_paragraphs.shape)  # [1, 30, 30, 768]
        # print(tail_paragraphs.shape)  # [1, 30, 768]
        concated_paragraphs = torch.cat((previous_paragraphs, tail_paragraphs), dim=2)  # caution for the dim
        return concated_paragraphs##, mask
    
    def _sequence_align(self, head_paragraphs, previous_ids=None):
        """
        :param head_paragraphs: (batch, max_node_num, hidden_dim_for_head_representations) or (batch, hidden_dim_for_head_representations)
        :param tail_paragraphs: (batch, max_node_num, hidden_dim_for_tail_representations) or (batch, hidden_dim_for_tail_representations)
        :param previous_ids: (batch, max_node_num), indicating for each tail paragraph its corresponding head paragraph index, 0 means dummy node and padding!!!
        if want to predict each pair incrementally, you can just input max_node_num=1 and previous_ids=None
        """

        if len(head_paragraphs.shape) == 2:  # TODO
            assert len(tail_paragraphs.shape) == 2
            batch_size, head_hidden_dim = head_paragraphs.shape
            batch_size, tail_hidden_dim = tail_paragraphs.shape
        elif len(head_paragraphs.shape) == 3:
            assert len(tail_paragraphs.shape) == 3
            batch_size, max_node_num, head_hidden_dim = head_paragraphs.shape
            batch_size, max_node_num, tail_hidden_dim = tail_paragraphs.shape
        # TODO: we want the concatenation progress to be parallel, i.e. do not use for loop as possible
        ##mask = (previous_ids == 0).long()
        # print(previous_ids.shape)  # [1, 30]
        # print(previous_ids)
        # previous_paragraphs = paragraphs[:][previous_ids][:]
        if previous_ids is None:
            previous_ids = torch.tensor([list(range(max_node_num))], dtype=torch.int).to(head_paragraphs.device)  # TODO: batchify implementation
        # print("head_paragraphs_vector", head_paragraphs[:,:,:5])
        previous_paragraphs = head_paragraphs[:, previous_ids.squeeze(0), :]
        # print("previous_paragraphs_vector", previous_paragraphs[:,:,:5])
        # print(previous_ids)
        # print(previous_paragraphs.shape)  # [1, 30, 30, 768]
        # print(tail_paragraphs.shape)  # [1, 30, 768]
        return previous_paragraphs

    

    def _get_relative_position_encoding_for_tail(self, paragraph_encoding, tail_index):
        batch_size, num_paragraphs = paragraph_encoding.shape[:2]

        relative_positions = list(range(tail_index, 0, -1))
        relative_positions = relative_positions + [0] * (num_paragraphs - len(relative_positions))  # TODO: 0 stands for invalid father node (current node or the node after current node)
        relative_positions = [relative_positions for n in range(batch_size)]  # batchify
        relative_positions = torch.tensor(relative_positions, dtype=torch.long).to(paragraph_encoding.device)
        
        relative_position_encodings = self.relative_position_encoding(relative_positions)  # (num_paragraph, 200)
        # print(f"Shape of relative_position_encoding = {relative_position_encodings.shape}")
        return relative_position_encodings
    
    def _get_relative_position_encoding(self, previous_ids):
        """
        : params previous_ids: (batch_size, num_paragraphs), 0 is for dummy head node or padding tail node
        """
        # relative_positions = list(range(tail_index, 0, -1))
        # relative_positions = relative_positions + [0] * (num_paragraphs - len(relative_positions))  # TODO: 0 stands for invalid father node (current node or the node after current node)
        # relative_positions = [relative_positions for n in range(batch_size)]  # batchify
        # relative_positions = torch.tensor(relative_positions, dtype=torch.long).to(paragraph_encoding.device)
        batch_size, num_paragraphs = previous_ids.shape
        tail_ids = torch.arange(num_paragraphs).expand(batch_size, num_paragraphs).to(previous_ids.device)
        relative_positions = tail_ids - previous_ids
        relative_position_encodings = self.relative_position_encoding(relative_positions)  # (num_paragraph, 200)
        # print(f"Shape of relative_position_encoding = {relative_position_encodings.shape}")
        return relative_position_encodings

    ##########
    # def _init_local_pairwise_combine_method(self, combine_method, bilinear_combine="tail"):
    def _init_pairwise_interaction_method(self, method, bilinear_combine="tail"):
        """
        combine_method is a `required` parameter ("None" is not allowed)
        bilinear_combine : ["head", "tail", "both"], indicating how to combine relative positional encodings with paragraph encodings in Bilinear case
        """
        namecards = {
            "bilinear_head": "pairwise_interaction_method = Bilinear(tail_encoding, relative_position_embedding(+)head_encoding)",
            "bilinear_tail": "pairwise_interaction_method = Bilinear(head_encoding, relative_position_embedding(+)tail_encoding)",
            "bilinear_both": f"pairwise_interaction_method = Bilinear(relative_position_embedding(+)head_encoding, relative_position_embedding(+)tail_encoding)",
            # "biaffine_head": "pairwise_interaction_method = BiAffine(tail_encoding, relative_position_embedding(+)head_encoding)"
            # "biaffine_tail": "pairwise_interaction_method = BiAffine(head_encoding, relative_position_embedding(+)tail_encoding)",
            # "biaffine_both": f"pairwise_interaction_method = BiAffine(relative_position_embedding(+)head_encoding, relative_position_embedding(+)tail_encoding)",
            "biaffine": "pairwise_interaction_method = BiAffine(head_encoding, relative_position_embedding(+)tail_encoding)",
            "variable-class-biaffine": "pairwise_interaction_method = Variable-Class-BiAffine(head_encoding(+)relative_position_embedding, tail_encoding)",
            "concate-linear": f"pairwise_interaction_method = Linear(head_encoding(+)relative_position_embedding(+)tail_encoding)",
            "concate-mlp": f"pairwise_interaction_method = MLP(head_encoding(+)relative_position_embedding(+)tail_encoding)",
        }
        print(namecards[method+"_"+bilinear_combine] if method in ["bilinear",] else namecards[method])

        if method == "bilinear":
            if bilinear_combine == "head":
                self.bilinear = nn.Bilinear(self.tail_input_dims, self.head_input_dims + self.relative_position_encoding_dims, self.output_dims)

                def concate_head_bilinear(head_paragraphs, tail_paragraphs, relative_position_encoding):
                    concated_tensor = torch.cat((head_paragraphs, relative_position_encoding), -1)
                    # print(f"Shape of concated_tensor = {concated_tensor.shape}")
                    return self.bilinear(tail_paragraphs, concated_tensor)
                self._local_pairwise_combine_method = concate_head_bilinear
            
            elif bilinear_combine == "tail":
                self.bilinear = nn.Bilinear(self.head_input_dims, self.tail_input_dims + self.relative_position_encoding_dims, self.output_dims)

                def concate_tail_bilinear(head_paragraphs, tail_paragraphs, relative_position_encoding):
                    concated_tensor = torch.cat((tail_paragraphs, relative_position_encoding), -1)
                    # print(f"Shape of concated_tensor = {concated_tensor.shape}")
                    return self.bilinear(head_paragraphs, concated_tensor)
                self._local_pairwise_combine_method = concate_tail_bilinear
            
            else:  # both
                self.bilinear = nn.Bilinear(self.head_input_dims + self.relative_position_encoding_dims, self.tail_input_dims + self.relative_position_encoding_dims, 1)

                def concate_both_bilinear(head_paragraphs, tail_paragraphs, relative_position_encoding):
                    concated_head_tensor = torch.cat((head_paragraphs, relative_position_encoding), -1)
                    concated_tail_tensor = torch.cat((tail_paragraphs, relative_position_encoding), -1)
                    # print(f"Shape of concated_head_tensor = {concated_head_tensor.shape}")
                    # print(f"Shape of concated_tail_tensor = {concated_tail_tensor.shape}")
                    return self.bilinear(concated_head_tensor, concated_tail_tensor)
                self._local_pairwise_combine_method = concate_both_bilinear
            # self.local_pairwise_combine_method = 

        elif method == "biaffine":
            self.biaffine = BiaffineAttention(self.head_input_dims, self.tail_input_dims + self.relative_position_encoding_dims, self.output_dims)
            def concate_tail_biaffine(head_paragraphs, tail_paragraphs, relative_position_encoding):
                concated_tensor = torch.cat((tail_paragraphs, relative_position_encoding), -1)
                return self.biaffine(head_paragraphs, concated_tensor)
            self._local_pairwise_combine_method = concate_tail_biaffine
        
        elif method == "variable-class-biaffine":  # 12/13
            self.biaffine = BiaffinePointerAttention(self.head_input_dims + self.relative_position_encoding_dims, self.tail_input_dims, self.output_dims)
            def concate_tail_biaffine(head_paragraphs, tail_paragraphs, relative_position_encoding):
                concated_tensor = torch.cat((head_paragraphs, relative_position_encoding), -1)
                return self.biaffine(concated_tensor, tail_paragraphs)
            self._local_pairwise_combine_method = concate_tail_biaffine

        elif method == "concate-linear":
            self.linear = nn.Linear(self.head_input_dims + self.tail_input_dims + self.relative_position_encoding_dims, self.output_dims)
            # self.pairwise_linear_interaction = nn.Linear(2*self.paragraph_vector_dim + self.relative_position_encoding_dim, 1)

            def concate_linear(head_paragraphs, tail_paragraphs, relative_position_encoding):
                concated_paragraphs = torch.cat((head_paragraphs, tail_paragraphs, relative_position_encoding), dim=2)
                # concated_paragraphs = self._concate(head_paragraphs, tail_paragraphs, previous_ids)  # (batch_size, num_paragraphs, head_hidden_dim+tail_hidden_dim)
                logits = self.mlp(concated_paragraphs)
                # concated_tensor = torch.cat((head_encoding, tail_encoding, relative_position_encoding), -1)
                # return self.pairwise_linear_interaction(concated_tensor)
                return logits
            self._local_pairwise_combine_method = concate_linear

        elif method == "concate-mlp":
            concate_input_dims = self.head_input_dims + self.tail_input_dims + self.relative_position_encoding_dims
            self.mlp = nn.Sequential(nn.Linear(concate_input_dims, concate_input_dims),
                                     nn.Tanh(),
                                     nn.Linear(concate_input_dims, self.output_dims),
                                     )
            # self.pairwise_mlp_interaction = nn.Sequential(nn.Linear(2*self.paragraph_vector_dim + self.relative_position_encoding_dim, self.input_dims),
            #                                               nn.ReLU(),
            #                                               nn.Linear(self.input_dims, 1),
            #                                               nn.Sigmoid()
            #                                               )

            def concate_mlp(head_paragraphs, tail_paragraphs, relative_position_encoding):
                concated_paragraphs = torch.cat((head_paragraphs, tail_paragraphs, relative_position_encoding), dim=2)
                # concated_paragraphs = self._concate(head_paragraphs, tail_paragraphs, previous_ids)  # (batch_size, num_paragraphs, head_hidden_dim+tail_hidden_dim)
                logits = self.mlp(concated_paragraphs)  # (batch_size, num_paragraphs, output_dims)
                # concated_tensor = torch.cat((head_encoding, tail_encoding, relative_position_encoding), -1)
                # return self.pairwise_mlp_interaction(concated_tensor)
                return logits
            self._local_pairwise_combine_method = concate_mlp

        # elif combine_method == "trilinear":
        #     # self.pairwise_trilinear_interaction = 
        #     pass


    # def _combine_local_pairwise_information(self, head_paragraphs, tail_paragraphs, previous_ids=None,):
    def _pairwise_interaction(self, head_paragraphs, tail_paragraphs, previous_ids=None,):
        batch_size, num_paragraphs = tail_paragraphs.shape[:2]
        # tails = tail_vectors[:,tail_index,:].repeat(1, num_paragraphs, 1)  # (Batch, num_paragraph, hidden_dim)
        relative_position_encoding = self._get_relative_position_encoding(previous_ids=previous_ids) if self.relative_position_encoding else torch.tensor([[[]]*num_paragraphs]*batch_size).to(head_paragraphs.device)
        # print(f"Shape of head_vectors = {head_vectors.shape}")
        # print(f"Shape of tails = {tails.shape}")
        # print(f"Shape of relative_position_encoding = {relative_position_encoding.shape}")
        logits = self._local_pairwise_combine_method(head_paragraphs, tail_paragraphs, relative_position_encoding)  # (Batch, num_paragraph, 1) -> (Batch, 1, num_paragraph)
        # print(f"Shape of father_node_logit_vector = {father_node_logit_vector.shape}")
        return logits
    
    def _pairwise_interaction_one_to_all(self, head_paragraphs, tail_paragraphs, tail_index,):
        batch_size, num_paragraphs = tail_paragraphs.shape[:2]
        tails = tail_paragraphs[:,tail_index,:].repeat(1, num_paragraphs, 1)  # (Batch, num_paragraph, hidden_dim)
        relative_position_encoding = self._get_relative_position_encoding_for_tail(tail_paragraphs, tail_index=tail_index) if self.relative_position_encoding else torch.tensor([[[]]*num_paragraphs]*batch_size).to(head_paragraphs.device)
        # print(f"Shape of head_vectors = {head_vectors.shape}")
        # print(f"Shape of tails = {tails.shape}")
        # print(f"Shape of relative_position_encoding = {relative_position_encoding.shape}")
        father_node_logit_vector = self._local_pairwise_combine_method(head_paragraphs, tails, relative_position_encoding).permute(0,2,1)  # (Batch, num_paragraph, 1) -> (Batch, 1, num_paragraph)
        # print(f"Shape of father_node_logit_vector = {father_node_logit_vector.shape}")
        return father_node_logit_vector

    def _pairwise_interaction_graph_one_to_all(self, head_to_tail_paragraphs, tail_to_head_paragraphs, tail_index,):
        # 12/31, for forward_with_graph_encodings()
        """
        head_to_tail_paragraphs: (Batch_size, num_paragraph, hidden_dim)
        tail_to_head_paragraphs: (Batch_size, num_paragraph, hidden_dim)
        """

        batch_size, num_paragraphs = tail_to_head_paragraphs.shape[:2]
        """tails = tail_paragraphs[:,tail_index,:].repeat(1, num_paragraphs, 1)  # (Batch, num_paragraph, hidden_dim)"""
        # print(self.relative_position_encoding)
        relative_position_encoding = self._get_relative_position_encoding_for_tail(tail_to_head_paragraphs, tail_index=tail_index) if self.relative_position_encoding else torch.tensor([[[]]*num_paragraphs]*batch_size).to(head_to_tail_paragraphs.device)
        # print(f"Shape of head_vectors = {head_vectors.shape}")
        # print(f"Shape of tails = {tails.shape}")
        # print(f"Shape of relative_position_encoding = {relative_position_encoding.shape}")
        father_node_logit_vector = self._local_pairwise_combine_method(head_to_tail_paragraphs, tail_to_head_paragraphs, relative_position_encoding).permute(0,2,1)  # (Batch, num_paragraph, 1) -> (Batch, 1, num_paragraph)
        # print(f"Shape of father_node_logit_vector = {father_node_logit_vector.shape}")
        # print("father_node_logit_vector", father_node_logit_vector.shape)
        return father_node_logit_vector
    
    
    def forward_with_graph_encodings(self, directed_graph_encodings, golden_parent_ids=None, golden_parent_labels=None, pad_index=-100):
        """
        Doing masked self attention
        :param directed_graph_encodings: (batch, max_node_num+1, max_node_num+1, self.input_dims=2*params.path_hidden_size)
        :param previous_ids: (batch_size, max_node_num+1, 1),
        :param golden: (batch_size, max_node_num)
        :return: previous_node_logits: (batch_size, num_paragraphs, 3)
        """
        loss = None

        batch_size, num_paragraphs = directed_graph_encodings.shape[:2]
        
        directed_graph_encodings_head_to_tail = directed_graph_encodings
        directed_graph_encodings_tail_to_head = directed_graph_encodings.transpose(1, 2)
        directed_graph_encodings_head_to_tail = self.head_mlp(directed_graph_encodings_head_to_tail)  # (Batch, num_paragraph, hidden_dim) # 11/30
        directed_graph_encodings_tail_to_head = self.tail_mlp(directed_graph_encodings_tail_to_head)  # (Batch, num_paragraph, hidden_dim) # 11/30

        """concated_paragraphs = self._concate(head_paragraphs, tail_paragraphs, previous_ids)  # (batch_size, num_paragraphs, head_hidden_dim+tail_hidden_dim)
        logits = self.mlp(concated_paragraphs)  # (batch_size, num_paragraphs, output_dims)"""
        # head_paragraphs_aligned = self._sequence_align(head_paragraphs, previous_ids)
        # logits = self._pairwise_interaction(head_paragraphs_aligned, tail_paragraphs, previous_ids)

        ## pointer interaction ##
        father_node_logit_matrix = torch.zeros(batch_size, num_paragraphs, num_paragraphs).to(directed_graph_encodings.device)  # 11/30 self.output_dims

        for t in range(num_paragraphs):  # for loop is for avoiding out of memory
            # print("father_node_logit_matrix[:,t,:]", father_node_logit_matrix[:,t,:].shape)
            father_node_logit_matrix[:,t,:] = self._pairwise_interaction_graph_one_to_all(
                directed_graph_encodings_head_to_tail[:,:,t],
                directed_graph_encodings_tail_to_head[:,:,t],
                tail_index=t
            ).squeeze(1)
                
        attn_mask = (torch.ones(num_paragraphs, num_paragraphs) - torch.tril(torch.ones(num_paragraphs, num_paragraphs))).to(paragraphs.device)  # here, num_paragraphs is num edus + dummy node
        # father_node_logit_matrix = father_node_logit_matrix + (1 - attn_mask + (-10e9) * attn_mask)  # 1106  # 1121
        father_node_logit_matrix = father_node_logit_matrix + ((-10e9) * attn_mask)
        father_node_logit_scores = torch.softmax(father_node_logit_matrix[:, 1:, :], dim=2)  # wipe off dummy node, but keep the sum of probabilities = 1

        # mask for learning
        if golden_parent_ids is not None:  # 11/21 ???
            mask = golden_parent_ids.clone()
            mask[mask != pad_index] = 1
            mask[mask == pad_index] = 0
            mask = mask.detach()
        else:
            mask = torch.ones(head_paragraphs.shape[0], head_paragraphs.shape[1]).to(dtype=torch.bool, device=head_paragraphs.device)
        """mask = mask[:,1:]  # wipe off dummy node""" 
        
        # train
        if golden_parent_ids is not None:
            golden_parent_ids = golden_parent_ids[mask != 0]
            golden_parent_ids = golden_parent_ids + 1
            masked_logits = father_node_logit_matrix[:, 1:, :][mask != 0]
            if self.loss_type == "ce":
                loss = self.loss(masked_logits, golden_parent_ids)
                # print(flatten_father_node_logit_matrix)
                # print(flatten_golden)
                # print(loss)
            else:
                masked_father_node_logit_scores = father_node_logit_scores[mask != 0]
                masked_father_node_logit_scores = torch.flatten(masked_father_node_logit_scores, end_dim=1)
                loss = self.loss(masked_father_node_logit_scores, golden_parent_ids)
        
        outputs = (father_node_logit_scores, loss)
        return outputs
    

    def forward_with_graph_encodings_return_logits(self, directed_graph_encodings,):
        # 12/31 This is called by forward() of SSA
        """
        Doing masked self attention
        :param directed_graph_encodings: (batch, max_node_num+1, max_node_num+1, self.input_dims=2*params.path_hidden_size)
        :param previous_ids: (batch_size, max_node_num+1, 1),
        :param golden: (batch_size, max_node_num)
        :return: previous_node_logits: (batch_size, num_paragraphs, 3)
        """
        loss = None

        batch_size, num_paragraphs = directed_graph_encodings.shape[:2]
        
        directed_graph_encodings_head_to_tail = directed_graph_encodings
        directed_graph_encodings_tail_to_head = directed_graph_encodings.transpose(1, 2)
        directed_graph_encodings_head_to_tail = self.head_mlp(directed_graph_encodings_head_to_tail)  # (Batch, num_paragraph, hidden_dim) # 11/30
        directed_graph_encodings_tail_to_head = self.tail_mlp(directed_graph_encodings_tail_to_head)  # (Batch, num_paragraph, hidden_dim) # 11/30

        """concated_paragraphs = self._concate(head_paragraphs, tail_paragraphs, previous_ids)  # (batch_size, num_paragraphs, head_hidden_dim+tail_hidden_dim)
        logits = self.mlp(concated_paragraphs)  # (batch_size, num_paragraphs, output_dims)"""
        # head_paragraphs_aligned = self._sequence_align(head_paragraphs, previous_ids)
        # logits = self._pairwise_interaction(head_paragraphs_aligned, tail_paragraphs, previous_ids)

        ## pointer interaction ##
        father_node_logit_matrix = torch.zeros(batch_size, num_paragraphs, num_paragraphs).to(directed_graph_encodings.device)  # 11/30 self.output_dims

        for t in range(num_paragraphs):  # for loop is for avoiding out of memory
            # print("father_node_logit_matrix[:,t,:]", father_node_logit_matrix[:,t,:].shape)
            father_node_logit_matrix[:,t,:] = self._pairwise_interaction_graph_one_to_all(
                directed_graph_encodings_head_to_tail[:,:,t],
                directed_graph_encodings_tail_to_head[:,:,t],
                tail_index=t
            ).squeeze(1)
                
        return father_node_logit_matrix

    
    def forward(self, head_paragraphs, tail_paragraphs, golden_parent_ids=None, golden_parent_labels=None, pad_index=-100):
        """
        Doing masked self attention
        :param paragraphs: (batch_size, num_paragraphs, hidden_dim)
        :param previous_ids: (batch_size, num_paragraphs), 0 is for dummy head node or padding tail node
        :param golden: (batch_size, num_paragraphs) if self.output_dims=1;   # TODO
        :return: previous_node_logits: (batch_size, num_paragraphs, output_dims)
        """
        loss = None

        batch_size, num_paragraphs = tail_paragraphs.shape[:2]
        head_paragraphs = self.head_mlp(head_paragraphs)  # (Batch, num_paragraph, hidden_dim) # 11/30
        tail_paragraphs = self.tail_mlp(tail_paragraphs)  # (Batch, num_paragraph, hidden_dim) # 11/30

        """concated_paragraphs = self._concate(head_paragraphs, tail_paragraphs, previous_ids)  # (batch_size, num_paragraphs, head_hidden_dim+tail_hidden_dim)
        logits = self.mlp(concated_paragraphs)  # (batch_size, num_paragraphs, output_dims)"""
        # head_paragraphs_aligned = self._sequence_align(head_paragraphs, previous_ids)
        # logits = self._pairwise_interaction(head_paragraphs_aligned, tail_paragraphs, previous_ids)

        ## pointer interaction ##
        father_node_logit_matrix = torch.zeros(batch_size, num_paragraphs, num_paragraphs).to(head_paragraphs.device)  # 11/30 self.output_dims

        for t in range(num_paragraphs):
            father_node_logit_matrix[:,t,:] = self._pairwise_interaction_one_to_all(head_paragraphs, tail_paragraphs, tail_index=t).squeeze(1)
                
        attn_mask = (torch.ones(num_paragraphs, num_paragraphs) - torch.tril(torch.ones(num_paragraphs, num_paragraphs))).to(head_paragraphs.device)  # here, num_paragraphs is num edus + dummy node
        # father_node_logit_matrix = father_node_logit_matrix + (1 - attn_mask + (-10e9) * attn_mask)  # 1106  # 1121
        father_node_logit_matrix = father_node_logit_matrix + ((-10e9) * attn_mask)
        father_node_logit_scores = torch.softmax(father_node_logit_matrix[:, 1:, :], dim=2)  # wipe off dummy node, but keep the sum of probabilities = 1
        
        
        # mask for learning
        if golden_parent_ids is not None:  # 11/21 ???
            mask = golden_parent_ids.clone()
            mask[mask != pad_index] = 1
            mask[mask == pad_index] = 0
            mask = mask.detach()
        else:
            mask = torch.ones(head_paragraphs.shape[0], head_paragraphs.shape[1]).to(dtype=torch.bool, device=head_paragraphs.device)
        """mask = mask[:,1:]  # wipe off dummy node"""
        
        # train
        if golden_parent_ids is not None:
            golden_parent_ids = golden_parent_ids[mask != 0]
            golden_parent_ids = golden_parent_ids + 1
            masked_logits = father_node_logit_matrix[:, 1:, :][mask != 0]
            if self.loss_type == "ce":
                loss = self.loss(masked_logits, golden_parent_ids)
                # print(flatten_father_node_logit_matrix)
                # print(flatten_golden)
                # print(loss)
            else:
                masked_father_node_logit_scores = father_node_logit_scores[mask != 0]
                masked_father_node_logit_scores = torch.flatten(masked_father_node_logit_scores, end_dim=1)
                loss = self.loss(masked_father_node_logit_scores, golden_parent_ids)
    
        # if golden is not None:
        #     # Compute loss
        #     flatten_golden = torch.flatten(golden)
        #     flatten_golden = flatten_golden + 1  # Deal with "NA" type
        #     flatten_father_node_logit_scores = torch.flatten(father_node_logit_scores, end_dim=1)
        #     flatten_father_node_logit_matrix = torch.flatten(father_node_logit_matrix[:, 1:, :], end_dim=1)  # TODO
        #     # print(flatten_father_node_logit_scores.shape)
        #     # print(flatten_golden)
        #     # print(flatten_golden.shape)
        #     # print(flatten_father_node_logit_matrix.shape)
        #     # print(flatten_father_node_logit_scores)
        #     # loss = self.loss(flatten_father_node_logit_scores, flatten_golden)
        #     if self.loss_type == "ce":
        #         loss = self.loss(flatten_father_node_logit_matrix, flatten_golden)
        #         # print(flatten_father_node_logit_matrix)
        #         # print(flatten_golden)
        #         # print(loss)
        #     else:
        #         loss = self.loss(flatten_father_node_logit_scores, flatten_golden)

        outputs = (father_node_logit_scores, loss)
        return outputs



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
        
        """self.relative_position_encoding = nn.Embedding(max_paragraph_num, max_paragraph_num)  # TODO: 0703
        # self.position_encoding = nn.Embedding(max_paragraph_num, input_dims)  # TODO: 0719
        onehot_weight = torch.FloatTensor(torch.eye(max_paragraph_num))  # TODO: 0722
        self.position_encoding = nn.Embedding.from_pretrained(onehot_weight)  # TODO: 0722"""
        
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
            onehot_weight = torch.FloatTensor(torch.eye(self.max_paragraph_num))  # TODO: 0722
            self.position_encoding = nn.Embedding.from_pretrained(onehot_weight)  # TODO: 0722
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
            self.position_encoding_combine_method = lambda x, y: x

    def _combine_position_encoding(self, paragraph_encoding,):
        if self.position_encoding is None:
            return paragraph_encoding
        batch_size = paragraph_encoding.shape[0]
        num_paragraphs = paragraph_encoding.shape[1]
        positions = list(range(num_paragraphs))  # TODO: 0721
        positions = [positions for n in range(batch_size)]  # TODO: 0721
        positions = torch.tensor(positions, dtype=torch.long).to(paragraph_encoding.device)  # TODO: 0721
        paragraph_encoding = self.position_encoding_combine_method(paragraph_encoding, self.position_encoding(positions))  # TODO: 0721
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
            onehot_weight = torch.FloatTensor(torch.eye(self.max_paragraph_num))  # TODO: 0722
            self.relative_position_encoding = nn.Embedding.from_pretrained(onehot_weight)  # TODO: 0722
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

        elif combine_method == "linear":
            print(f"local_pairwise_combine_method = Linear(head_encoding(+)relative_position_embedding(+)tail_encoding)")
            self.pairwise_linear_interaction = nn.Linear(2*self.paragraph_vector_dim + self.relative_position_encoding_dim, 1)

            def concate_linear(head_encoding, tail_encoding, relative_position_encoding):
                concated_tensor = torch.cat((head_encoding, tail_encoding, relative_position_encoding), -1)
                return self.pairwise_linear_interaction(concated_tensor)
            self.local_pairwise_combine_method = concate_linear

        elif combine_method == "mlp":
            print(f"local_pairwise_combine_method = MLP(head_encoding(+)relative_position_embedding(+)tail_encoding)")
            self.pairwise_mlp_interaction = nn.Sequential(nn.Linear(2*self.paragraph_vector_dim + self.relative_position_encoding_dim, self.input_dims),
                                                          nn.ReLU(),
                                                          nn.Linear(self.input_dims, 1),
                                                          nn.Sigmoid()
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


        father_node_logit_matrix = torch.zeros(batch_size, num_paragraphs, num_paragraphs).to(paragraphs.device)

        for t in range(num_paragraphs):
            father_node_logit_matrix[:,t,:] = self._combine_local_pairwise_information(head_vectors, tail_vectors, tail_index=t)
                
        attn_mask = (torch.ones(num_paragraphs, num_paragraphs) - torch.tril(torch.ones(num_paragraphs, num_paragraphs))).to(paragraphs.device)  # here, num_paragraphs is num edus + dummy node
        # father_node_logit_matrix = father_node_logit_matrix + (1 - attn_mask + (-10e9) * attn_mask)  # 1106  # 1121
        father_node_logit_matrix = father_node_logit_matrix + ((-10e9) * attn_mask)


        # logits = self.bilinear(head_vectors, tail_vectors)
        """inf_mask = (1 - padding) * (-1.0e10)
        print(father_node_logit_matrix)
        father_node_logit_matrix = inf_mask * father_node_logit_matrix
        print(father_node_logit_matrix)""" # Do not need this padding (any more because batch_size=1 and the padding is by nature a triangle padding)
        # print(father_node_logit_matrix.shape)
        # father_node_logit_scores = torch.softmax(father_node_logit_matrix, dim=2)[:, 1:, :]  # wipe off dummy node
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


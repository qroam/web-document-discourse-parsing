# -*- coding: utf-8 -*-
from typing import List
import torch
from torch import nn
from collections import defaultdict


class GraphClassifier(nn.Module):
    """
    Predicting the Continuity ["Continue", "Break", "Combine"(Optional)] between same-level adjancent paragraphs
    Mechanism: Doing Concat(head, tail) -> Linear() to projection the into label space
    """
    def __init__(self, input_dims, output_dims, combine_before=False, position_highlight_mlp=False,):
        super().__init__()
        self.input_dims = input_dims
        # self.output_dims = 2 if combine_before else 3  # TODO 0729
        self.output_dims = output_dims  # 1213
        self.mlp = nn.Sequential(nn.Linear(input_dims, input_dims),
                                 nn.ReLU(),
                                 nn.Linear(input_dims, self.output_dims),  # 3 for ["Break", "Continue", "Combine"]
                                #  nn.Tanh()  # 12/14
                                 )
        """self.head_mlp = nn.Sequential(nn.Linear(head_input_dims, head_input_dims),
                                      nn.ReLU(),
                                      nn.Linear(head_input_dims, head_input_dims),
                                      nn.Tanh()
                                      ) if position_highlight_mlp else lambda x: x  # added 12/26
        self.tail_mlp = nn.Sequential(nn.Linear(tail_input_dims, tail_input_dims),
                                      nn.ReLU(),
                                      nn.Linear(tail_input_dims, tail_input_dims),
                                      nn.Tanh()
                                      ) if position_highlight_mlp else lambda x: x  # added 12/26"""
        self.loss = nn.CrossEntropyLoss()

        print("initialized GraphClassifier")


    def forward(self, directed_graph_encodings, previous_ids, golden=None):
        """
        Doing masked self attention
        :param directed_graph_encodings: (batch, max_node_num+1, max_node_num+1, self.input_dims=2*params.path_hidden_size)
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
        pair_node_encodings = torch.zeros(batch_size, max_node_num, hidden_dim).to(directed_graph_encodings.device)
        for i in range(previous_ids.shape[0]):
            for j in range(previous_ids.shape[1]):
                pair_node_encodings[i, j] = directed_graph_encodings[i, j, previous_ids[i][j]]
        # directed_graph_encodings[:, :, previous_ids, :]
        # print(pair_node_encodings.shape)
        logits = self.mlp(pair_node_encodings)
        # print(previous_ids)
        # print(previous_ids.shape)
        previous_mask = previous_ids.clone()
        previous_mask[previous_mask != 0] = 1
        previous_mask = previous_mask.detach()

        previous_mask = previous_mask[:,1:]  # wipe off dummy node
        logits = logits[:,1:,:]  # wipe off dummy node
        previous_relation_scores = torch.softmax(logits, dim=2)
        # print(golden.shape)
        # print(logits.shape)
        # print(previous_mask.shape)

        if golden is not None:
            golden = golden[previous_mask != 0]
            logits = logits[previous_mask != 0]
        
            loss = self.loss(logits, golden)

        # print(previous_relation_scores)
        outputs = (previous_relation_scores, loss)
        return outputs



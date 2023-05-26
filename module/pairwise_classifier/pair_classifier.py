# -*- coding: utf-8 -*-
from typing import List
import torch
from torch import nn
from collections import defaultdict


class PairClassifier(nn.Module):
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
        self.head_mlp = nn.Sequential(nn.Linear(input_dims //2, input_dims //2),
                                      nn.ReLU(),
                                      nn.Linear(input_dims //2, input_dims //2),
                                      nn.Tanh()
                                      ) if position_highlight_mlp else lambda x: x  # added 12/26
        self.tail_mlp = nn.Sequential(nn.Linear(input_dims //2, input_dims //2),
                                      nn.ReLU(),
                                      nn.Linear(input_dims //2, input_dims //2),
                                      nn.Tanh()
                                      ) if position_highlight_mlp else lambda x: x  # added 12/26
        self.loss = nn.CrossEntropyLoss()

        print("initialized PairClassifier")

    def concate(self, paragraphs, previous_ids):
        # TODO: we want the concatenation progress to be parallel, i.e. do not use for loop as possible
        mask = (previous_ids == 0).long()
        # print(previous_ids.shape)  # [1, 30]
        # print(previous_ids)
        # previous_paragraphs = paragraphs[:][previous_ids][:]
        previous_paragraphs = paragraphs[:, previous_ids.squeeze(0), :]  # TODO batch implementation
        # print(previous_paragraphs.shape)  # [1, 30, 30, 768]
        # print(paragraphs.shape)  # [1, 30, 768]
        previous_paragraphs = self.head_mlp(previous_paragraphs)  # 12/26
        paragraphs = self.tail_mlp(paragraphs)  # 12/26
        concated_paragraphs = torch.cat((previous_paragraphs, paragraphs), dim=2)  # caution for the dim
        return concated_paragraphs, mask

    def forward(self, paragraphs, previous_ids, golden=None):
        """
        Doing masked self attention
        :param paragraphs: (batch_size, num_paragraphs, hidden_dim)
        :param previous_ids: (batch_size, num_paragraphs, 1),
        :param golden: (batch_size, num_paragraphs)
        :return: previous_node_logits: (batch_size, num_paragraphs, 3)
        """
        loss = None

        concated_paragraphs, mask = self.concate(paragraphs, previous_ids)  # (batch_size, num_paragraphs, 2 * hidden_dim)
        logits = self.mlp(concated_paragraphs)  # (batch_size, num_paragraphs, 3)
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

        outputs = (previous_relation_scores, loss)
        return outputs


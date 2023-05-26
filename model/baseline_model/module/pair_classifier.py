# -*- coding: utf-8 -*-
from typing import List
import torch
from torch import nn
from collections import defaultdict


def father_id_to_previous_id(father_ids: List):
    batch_size = len(father_ids)
    batch_previous_id_list = []

    for instance in father_ids:
        latest_id_dict = defaultdict(int)  # 记录属于同一个父节点的最后一个节点
        previous_id_list = [0]  # 记录每个节点的前继节点，0代表无前继节点

        for i, node_father in enumerate(instance):
            node_father = int(node_father)  # That is important, otherwise previous_id_list would become all zeors
            # print(latest_id_dict)
            # print(node_father)
            """if i == 0:  # Dummy node
                # previous_id_list.append(-1)
                previous_id_list.append(0)
                continue"""
            if latest_id_dict[node_father] == 0:
                # previous_id_list.append(-1)
                previous_id_list.append(0)  # the first child node, whose previous node is NA
                latest_id_dict[node_father] = i + 1
                
            else:
                previous_id_list.append(latest_id_dict[node_father])
                latest_id_dict[node_father] = i + 1
        batch_previous_id_list.append(previous_id_list)
    # return torch.tensor(previous_id_list).to(father_ids.device)
    return batch_previous_id_list


class Pair_Classifier(nn.Module):
    """
    Predicting the Continuity ["Continue", "Break", "Combine"(Optional)] between same-level adjancent paragraphs
    Mechanism: Doing Concat(head, tail) -> Linear() to projection the into label space
    """
    def __init__(self, input_dims, combine_before=False):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = 2 if combine_before else 3  # TODO 0729
        self.mlp = nn.Sequential(nn.Linear(input_dims, input_dims),
                                 nn.ReLU(),
                                 nn.Linear(input_dims, self.output_dims),  # 3 for ["Break", "Continue", "Combine"]
                                 nn.Tanh()
                                 )
        self.loss = nn.CrossEntropyLoss()

    def concate(self, paragraphs, previous_ids):
        # TODO: we want the concatenation progress to be parallel, i.e. do not use for loop as possible
        mask = (previous_ids == 0).long()
        # print(previous_ids.shape)  # [1, 30]
        # print(previous_ids)
        # previous_paragraphs = paragraphs[:][previous_ids][:]
        previous_paragraphs = paragraphs[:, previous_ids.squeeze(0), :]
        # print(previous_paragraphs.shape)  # [1, 30, 30, 768]
        # print(paragraphs.shape)  # [1, 30, 768]
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
            # golden = golden * previous_mask
            # TODO
            # logits[:][previous_mask == 0][:] = [1., 0., 0., 0.]  # Mask out the nodes which have no previous node in tree
            # print(golden.shape)
            # print(logits.shape)
            # print(previous_mask.shape)
            golden = golden[previous_mask != 0]
            logits = logits[previous_mask != 0]
            # print(golden.shape)
            # print(logits.shape)
            # golden[golden==3] = 0  # 类别数一致
            golden[golden==3] = 1  # TODO 0704

            # print(previous_mask)
            # print(golden)
            # print(logits)
            loss = self.loss(logits, golden)

        outputs = (previous_relation_scores, loss)
        return outputs



if __name__ == "__main__":
    print(father_id_to_previous_id([[0, 1, 2,2,2,2,1,7,7,7,7,7,1,13,13,13,13,13,13,13,0,0,0,0,0,0,]+[26]*15+[0,0]]))
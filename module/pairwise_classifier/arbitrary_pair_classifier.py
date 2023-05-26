# -*- coding: utf-8 -*-
from typing import List
import torch
from torch import nn
from collections import defaultdict


def father_id_to_previous_id(father_ids: List):
    batch_size = len(father_ids)
    batch_previous_id_list = []

    for instance in father_ids:
        latest_id_dict = defaultdict(int)
        previous_id_list = [0]

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


class ArbitraryPairClassifier(nn.Module):
    """
    Predicting the Continuity ["Continue", "Break", "Combine"(Optional)] between same-level adjancent paragraphs
    Mechanism: Doing Concat(head, tail) -> Linear() to projection the into label space
    """
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.mlp = nn.Sequential(nn.Linear(input_dims, input_dims),
                                 nn.Tanh(),
                                 nn.Linear(input_dims, output_dims),
                                 )
        self.loss = nn.CrossEntropyLoss()

    def concate(self, head_paragraphs, tail_paragraphs, previous_ids=None):
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

    def forward(self, head_paragraphs, tail_paragraphs, previous_ids=None, golden=None):
        """
        Doing masked self attention
        :param paragraphs: (batch_size, num_paragraphs, hidden_dim)
        :param previous_ids: (batch_size, num_paragraphs, 1), 0 is for dummy head node or padding tail node
        :param golden: (batch_size, num_paragraphs)
        :return: previous_node_logits: (batch_size, num_paragraphs, output_dims)
        """
        loss = None

        concated_paragraphs = self.concate(head_paragraphs, tail_paragraphs, previous_ids)  # (batch_size, num_paragraphs, head_hidden_dim+tail_hidden_dim)
        logits = self.mlp(concated_paragraphs)  # (batch_size, num_paragraphs, output_dims)
        # print(previous_ids)
        # print(previous_ids.shape)
        # if previous_ids is None:
        print("previous_ids:", previous_ids)
        print("golden:", golden)
        if previous_ids is not None:  # 11/21 ???
            previous_mask = previous_ids.clone()
            previous_mask[previous_mask != 0] = 1
            previous_mask = previous_mask.detach()
        else:
            previous_mask = torch.ones(head_paragraphs.shape[0], head_paragraphs.shape[1]).to(dtype=torch.bool, device=head_paragraphs.device)

        previous_mask = previous_mask[:,1:]  # wipe off dummy node
        logits = logits[:,1:,:]  # wipe off dummy node
        print("previous_mask:", previous_mask)
        print("logits:", logits)
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



if __name__ == "__main__":
    print(father_id_to_previous_id([[0, 1, 2,2,2,2,1,7,7,7,7,7,1,13,13,13,13,13,13,13,0,0,0,0,0,0,]+[26]*15+[0,0]]))
import copy
import json
import math
import pickle
import random
import re
from collections import Counter

# import nltk
import torch
import numpy as np
from sklearn.metrics import f1_score
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn.utils.rnn import pad_sequence

from utils import common_collate_fn

torch.autograd.set_detect_anomaly(True)



def nest_padding(sequence):
    max_cols = max([len(row) for batch in sequence for row in batch])
    max_rows = max([len(batch) for batch in sequence])
    sequence = [batch + [[0] * (max_cols)] * (max_rows - len(batch)) for batch in sequence]
    sequence = torch.tensor([row + [0] * (max_cols - len(row)) for batch in sequence for row in batch])
    return sequence.reshape(-1, max_rows, max_cols)


def eval_collate_fn(batch):
    output = common_collate_fn(batch)
    # texts, pairs, graphs, lengths, speakers, turns, edu_nums, ids = zip(*examples)
    # texts = DialogueDataset.nest_padding(texts)
    # texts = nest_padding(batch["input_ids"])
    # lengths = pad_sequence(batch["lengths"], batch_first=True, padding_value=1)  # nn.utils.rnn.pad_sequence
    # speakers = ints_to_tensor(list(batch["speakers"])) if "speakers" in batch.keys() else None
    # turns = ints_to_tensor(list(batch["turns"])) if "turns" in batch.keys() else None
    # graphs = ints_to_tensor(batch["graphs"])
    # edu_nums = torch.tensor(batch["edu_nums"])

    """texts = nest_padding([instance["input_ids"] for instance in batch])"""
    ## num_paragraph = len(batch[0]["input_ids"])

    ## max_len = max([len(paragraph) for instance in batch for paragraph in instance["input_ids"]])
    ## input_ids = [[paragraph + [0] * (max_len - len(paragraph)) for paragraph in instance["input_ids"]] for instance in
    ##              batch]
    ## input_mask = [[[1.0] * len(paragraph) + [0.0] * (max_len - len(paragraph)) for paragraph in instance["input_ids"]]
    ##               for instance in batch]
    ## input_ids = torch.tensor(input_ids, dtype=torch.long)
    ## input_mask = torch.tensor(input_mask, dtype=torch.float)
    
    # print([instance["lengths"] for instance in batch])
    lengths = pad_sequence([torch.tensor(instance["lengths"]) for instance in batch], batch_first=True, padding_value=1)  # nn.utils.rnn.pad_sequence
    speakers = ints_to_tensor([instance["speakers"] for instance in batch]) if "speakers" in batch[0].keys() else None
    turns = ints_to_tensor([instance["turns"] for instance in batch]) if "turns" in batch[0].keys() else None
    graphs = ints_to_tensor([instance["graphs"] for instance in batch])
    edu_nums = torch.tensor([instance["edu_nums"] for instance in batch])
    pairs = [instance["pairs"] for instance in batch]

    ## ids = [instance["id"] for instance in batch]
    ## node_modal = [instance["Node_modal"] for instance in batch]
    ## meta = {
    ##     "ids": ids,
    ##     "node_modal": node_modal,
    ##     "golden_parent_ids": [instance["Father"] for instance in batch],
    ##     "golden_parent_relations": [instance["Father_Relation"] for instance in batch],
    ##     "golden_previous_ids": [instance["Previous"] for instance in batch],
    ##     "golden_previous_relations": [instance["Previous_Relation"] for instance in batch],
    ## }
    # print("graphs", graphs)

    # previous_node_ids = father_id_to_previous_id([[idx + 1 for idx in father_ids] for father_ids in father_labels])
    """previous_ids = [instance["Previous"] for instance in batch]
    previous_ids = torch.tensor(previous_ids, dtype=torch.long)
    previous_labels = [instance["Previous_Relation_ids"] for instance in batch]
    previous_labels = [[idx if (idx!=3 or previous_node_ids[i][j]==0) else 1 for j, idx in enumerate(lst)] for i, lst in enumerate(previous_labels)]
    previous_labels = torch.tensor(previous_labels, dtype=torch.long)"""
    # previous_ids = [instance["Previous"] for instance in batch]
    ## previous_ids = [ [0] + [idx+1 for idx in instance["Previous"]] for instance in batch]
    ## previous_labels = [instance["Previous_Relation_ids"] for instance in batch]
    ## previous_labels = [[idx if (idx != 3 or previous_ids[i][j + 1] == 0) else 1 for j, idx in enumerate(lst)] for i, lst in enumerate(previous_labels)]
    ## previous_ids = torch.tensor(previous_ids, dtype=torch.long)
    ## previous_labels = torch.tensor(previous_labels, dtype=torch.long)
    

    # output = {"meta": meta,
    output.update( {
            ##   "input_ids": input_ids,
            ##   "input_mask": input_mask,
              # "input_ids": input_ids,
            #   '''"texts": texts,'''
              "graphs": graphs,
              # "input_mask": input_mask,
              # "padding": upper_triangluar_padding,
              # "golden_parent": father_labels,
              # "golden_previous_ids": previous_node_ids,
              # "golden_previous": previous_labels,
            ##   "golden_previous_ids": previous_ids,
            ##   "golden_previous_labels": previous_labels,
              "lengths": lengths,
              "edu_nums": edu_nums,
              "speakers": speakers,
              "turns": turns,
              "pairs": pairs,
              })
    # return texts, pairs, graphs, lengths, speakers, turns, edu_nums, ids
    return output



def pad_tensors(tensors):
    """
    Takes a list of `N` M-dimensional tensors (M<4) and returns a padded tensor.

    The padded tensor is `M+1` dimensional with size `N, S1, S2, ..., SM`
    where `Si` is the maximum value of dimension `i` amongst all tensors.
    """
    rep = tensors[0]
    padded_dim = []
    for dim in range(rep.dim()):
        max_dim = max([tensor.size(dim) for tensor in tensors])
        padded_dim.append(max_dim)
    padded_dim = [len(tensors)] + padded_dim
    padded_tensor = torch.zeros(padded_dim)
    padded_tensor = padded_tensor.type_as(rep)
    for i, tensor in enumerate(tensors):
        size = list(tensor.size())
        if len(size) == 1:
            padded_tensor[i, :size[0]] = tensor
        elif len(size) == 2:
            padded_tensor[i, :size[0], :size[1]] = tensor
        elif len(size) == 3:
            padded_tensor[i, :size[0], :size[1], :size[2]] = tensor
        else:
            raise ValueError('Padding is supported for upto 3D tensors at max.')
    return padded_tensor


def ints_to_tensor(ints):
    """
    Converts a nested list of integers to a padded tensor.
    """
    if isinstance(ints, torch.Tensor):
        return ints
    if isinstance(ints, list):
        if isinstance(ints[0], int):
            return torch.LongTensor(ints)
        if isinstance(ints[0], torch.Tensor):
            return pad_tensors(ints)
        if isinstance(ints[0], list):
            return ints_to_tensor([ints_to_tensor(inti) for inti in ints])


def get_mask(node_num, max_edu_dist):
    """
    node_num = edu_num + 1, (Batch_size)
    """
    batch_size, max_num = node_num.size(0), node_num.max()
    mask = torch.arange(max_num).unsqueeze(0).to(node_num.device) < node_num.unsqueeze(1)  # padding node mask in a batch, (max_node_num) -> (1, max_node_num)
    # node_num.unsqueeze(1): (Batch_size, 1), [[13], [52], [32], ...]
    # torch.arange(max_num).unsqueeze(0): (1, max_node_num), [[0, 1, 2, ..., max_num]]
    # Broadcast of "<"
    # -> mask: (Batch_size, max_node_num)

    mask = mask.unsqueeze(1).expand(batch_size, max_num, max_num)  # (Batch_size, 1 max_node_num) -> (Batch_size, max_node_num, max_node_num)
    mask = mask & mask.transpose(1,2)  # get the `square mask`, 1 for real nodes in each document in the batch, 0 for padding nodes
    mask = torch.tril(mask, -1)  # `triangular mask`, -1 means the diagnal line is not contained
    """
    0 0 0 
    1 0 0
    1 1 0
    """
    
    if max_num > max_edu_dist:
        mask = torch.triu(mask, max_edu_dist - max_num)  # dependencies larger than max_edu_dist is not considered
        """
        0 0 0 
        1 0 0
        0 1 0
        """
    return mask  # (Batch_size, max_node_num, max_node_num)


def compute_loss(link_scores, label_scores, graphs, mask, p=False, negative=False):
    #$$# for debug check
    #$$# print("[*] compute_loss(), called by Model._compute_loss()")
    """
    :param link_scores: (batch, max_node_num, max_node_num)
    :param label_scores: (batch, max_node_num, max_node_num, relation_type_num)
    :param graphs: (batch, num_node, num_node) 
    :param mask: (batch, max_node_num, max_node_num)
    the position be masked (0) is not considered in loss computation, whether it is correct or wrong edge
    """
    #$$# print(f"mask: {mask}, {mask.shape}")
    link_scores[~mask] = -1e9  # (batch, max_node_num, max_node_num). mask is a 0/1 tensor, where 0 means masked
    
    label_mask = (graphs!=0) & mask  # graphs is the golden labels. in graphs, 0 means No relation. label_mask is positions of gloden edges which has a label
    tmp_mask = (graphs.sum(-1)==0)&mask[:,:,0]
    link_mask = label_mask.clone()
    link_mask[:,:,0] = tmp_mask
    
    
    #$$# print(f"link_scores, after mask: {link_scores}, {link_scores.shape}")
    #$$# print(f"link_mask: label_mask = (graphs!=0) & mask -> link_mask[:,:,0] = tmp_mask: {link_mask}, {link_mask.shape}")
    link_scores = torch.nn.functional.softmax(link_scores, dim=-1)
    #$$# print(f"link_scores_softmax: {link_scores}, {link_scores.shape}")
    #$$# print(f"link_scores[link_mask], the final input to loss function: {link_scores[link_mask]}, {link_scores[link_mask].shape}")
    link_loss = -torch.log(link_scores[link_mask])  # NLLloss for link prediction. link_mask is positions of gloden edges 
    #$$# print(f"link_loss(nllloss): {link_loss}, {link_loss.shape}")

    relation_type_num = label_scores.size(-1)
    # print(label_scores)
    # print(label_mask)
    # print("graphs", graphs)
    # print("mask", mask)
    # print(label_scores[label_mask].reshape(-1, relation_type_num))
    # print(graphs[label_mask].reshape(-1))
    #$$# print("="*20)

    #$$# print(f"label_scores(logits): {label_scores}, {label_scores.shape}")
    #$$# print(f"label_mask = (graphs!=0) & mask: {label_mask}, {label_mask.shape}")
    #$$# print(f"graphs(golden label): {graphs}, {graphs.shape}")
    #$$# print(f"relation_type_num(the dim of label classification loss): {relation_type_num}")
    #$$# print(f"label_scores[label_mask], the final input (logits) to loss function: {label_scores[label_mask]}, {label_scores[label_mask].shape}")
    #$$# print(f"graphs[label_mask], the final input (golden) to loss function: {graphs[label_mask]}, {graphs[label_mask].shape}")
    label_loss = torch.nn.functional.cross_entropy(
        label_scores[label_mask].reshape(-1, relation_type_num),
        graphs[label_mask].reshape(-1),
        reduction='none'
    )  # CEloss for relation classification
    #$$# print(f"label_loss(celoss): {label_loss}, {label_loss.shape}")
    # print("aaaaa")
    # print(label_scores)
    
    if negative:
        #$$# print("="*20)
        # use negative loss, which drives model to predict "None_type" at the positions where no golden edges exist
        negative_mask = (graphs==0) & mask

        #$$# print(f"negative_mask: {negative_mask}, {negative_mask.shape}")
        # print(label_scores)
        # print(negative_mask)
        # print(label_scores.shape)
        # print(negative_mask.shape)
        # print(label_scores[negative_mask].reshape(-1, relation_type_num))
        # print(graphs[negative_mask].reshape(-1))
        #$$# print(f"label_scores[negative_mask], the final input (logits) to negative loss function: {label_scores[negative_mask]}, {label_scores[negative_mask].shape}")
        #$$# print(f"graphs[negative_mask], the final input (golden) to negative loss function: {graphs[negative_mask]}, {graphs[negative_mask].shape}")
    
        negative_loss = torch.nn.functional.cross_entropy(label_scores[negative_mask].reshape(-1, relation_type_num), graphs[negative_mask].reshape(-1), reduction='mean')
        return link_loss, label_loss, negative_loss
    if p:  # ???
        return link_loss, label_loss, torch.nn.functional.softmax(label_scores[label_mask],dim=-1)[torch.arange(label_scores[label_mask].size(0)),graphs[mask]]
    return link_loss, label_loss, None


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def record_eval_result(eval_matrix, predicted_result):
    for k, v in eval_matrix.items():
        if v is None:
            if isinstance(predicted_result[k], dict):
                eval_matrix[k] = [predicted_result[k]]
            else:
                eval_matrix[k] = predicted_result[k]
        elif isinstance(v, list):
            eval_matrix[k] += [predicted_result[k]]
        else:
            eval_matrix[k] = np.append(eval_matrix[k], predicted_result[k])


def get_error_statics(eval_matrix):
    # error type: 0 link error, 1 label error
    errors_0 = []
    errors_1 = []
    errors_dist_0 = [0] * 20
    errors_dist_1 = [0] * 20
    for hypothesis, reference in zip(eval_matrix['hypothesis'], eval_matrix['reference']):
        for h_p, h_r in hypothesis.items():
            h_x = h_p[0]
            h_y = h_p[1]
            for r_p, r_r in reference.items():
                r_x = r_p[0]
                r_y = r_p[1]
                if h_y == r_y and r_x < r_y:
                    if h_x == r_x and h_r != r_r:
                        errors_1.append((h_r, r_r))
                        errors_dist_1[h_y - h_x] += 1
                    elif h_x != r_x:
                        errors_0.append((h_r, r_r))
                        errors_dist_0[h_y - h_x] += 1
    return sorted(Counter(errors_0).items(), key=lambda x: x[1], reverse=True), sorted(Counter(errors_1).items(),
                                                                                       key=lambda x: x[1],
                                                                                       reverse=True), errors_dist_0, errors_dist_1

def survey(eval_matrix, id2types):
    survey_dict={}
    for hypothesis, reference in zip(eval_matrix['hypothesis'], eval_matrix['reference']):
        for pair in reference:
            label=reference[pair]
            if label not in survey_dict:
                survey_dict[label]=[0, 0, 1]
            else:
                survey_dict[label][2]+=1
            if pair in hypothesis:
                survey_dict[label][0]+=1
                if hypothesis[pair] == reference[pair]:
                    survey_dict[label][1]+=1
    for k, v in survey_dict.items():
        print(id2types[k], v[0], v[1], v[2], v[0]*1.0/v[2], v[1]*1.0/v[2])


def test_F1(eval_matrix):
    cnt_golden, cnt_pred, cnt_cor_bi, cnt_cor_multi = 0, 0, 0, 0
    for hypothesis, reference in zip(eval_matrix['hypothesis'], eval_matrix['reference']):
        cnt_golden += len(reference)
        for pair in hypothesis:
            if pair[0] != -1:
                cnt_pred += 1
                if pair in reference:
                    cnt_cor_bi += 1
                    if hypothesis[pair] == reference[pair]:
                        cnt_cor_multi += 1
    prec_bi, recall_bi = cnt_cor_bi * 1. / cnt_pred, cnt_cor_bi * 1. / cnt_golden
    f1_bi = 2 * prec_bi * recall_bi / (prec_bi + recall_bi)
    prec_multi, recall_multi = cnt_cor_multi * 1. / cnt_pred, cnt_cor_multi * 1. / cnt_golden
    f1_multi = 2 * prec_multi * recall_multi / (prec_multi + recall_multi)
    return f1_bi, f1_multi


def accuray_dist(eval_matrix):
    dist_sum=np.zeros(15)
    dist_yes=np.zeros(15)
    for hypothesis, reference in zip(eval_matrix['hypothesis'], eval_matrix['reference']):
        for pair in reference:
            dist_sum[pair[1]]+=1
            if pair in hypothesis and hypothesis[pair] == reference[pair]:
                dist_yes[pair[1]]+=1
    print(dist_yes/dist_sum)
    print(dist_sum)
    print(dist_yes.sum()/dist_sum.sum())


def tsinghua_F1(eval_matrix):
    cnt_golden, cnt_pred, cnt_cor_bi, cnt_cor_multi = 0, 0, 0, 0
    for hypothesis, reference, edu_num in zip(eval_matrix['hypothesis'], eval_matrix['reference'],
                                              eval_matrix['edu_num']):
        cnt = [0] * edu_num
        for r in reference:
            cnt[r[1]] += 1
        for i in range(edu_num):
            if cnt[i] == 0:
                cnt_golden += 1
        cnt_pred += 1
        if cnt[0] == 0:
            cnt_cor_bi += 1
            cnt_cor_multi += 1
        cnt_golden += len(reference)
        cnt_pred += len(hypothesis)
        for pair in hypothesis:
            if pair in reference:
                cnt_cor_bi += 1
                if hypothesis[pair] == reference[pair]:
                    cnt_cor_multi += 1
    prec_bi, recall_bi = cnt_cor_bi * 1. / cnt_pred, cnt_cor_bi * 1. / cnt_golden
    f1_bi = 2 * prec_bi * recall_bi / (prec_bi + recall_bi)
    prec_multi, recall_multi = cnt_cor_multi * 1. / cnt_pred, cnt_cor_multi * 1. / cnt_golden
    f1_multi = 2 * prec_multi * recall_multi / (prec_multi + recall_multi)
    return f1_bi, f1_multi

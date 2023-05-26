import copy
import json
import math
import pickle
import random
import re
from collections import Counter

import nltk
import torch
import numpy as np
from sklearn.metrics import f1_score
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn.utils.rnn import pad_sequence

from utils import common_collate_fn

torch.autograd.set_detect_anomaly(True)

def train_collate_fn(examples):
    def pool(d):
        d = sorted(d, key=lambda x: x[7])
        edu_nums = [x[7] for x in d]
        buckets = []
        i, j, t = 0, 0, 0
        for edu_num in edu_nums:
            if t + edu_num > args.batch_size:
                buckets.append((i, j))
                i, t = j, 0
            t += edu_num
            j += 1
        buckets.append((i, j))
        for bucket in buckets:
            batch = d[bucket[0]:bucket[1]]
            texts, sep_index, pairs, graphs, lengths, speakers, turns, edu_nums,\
            parsing_index, decoder_input, relation_labels, _ = zip(*batch)
            max_edu_seqlen = max(edu_nums)
            num_batch = len(batch)
            d_inputs = np.zeros([num_batch, max_edu_seqlen], dtype=np.compat.long)
            d_outputs = np.zeros([num_batch, max_edu_seqlen], dtype=np.compat.long)
            d_output_re = np.zeros([num_batch, max_edu_seqlen], dtype=np.compat.long)
            d_masks = np.zeros([num_batch, max_edu_seqlen, max_edu_seqlen + 1], dtype=np.uint8)
            for batchi,batch_example in enumerate(batch):
                _, _, _, _, _, _, _,_, parsing_index_, decoder_input_, relation_label_,_= batch_example
                for di, d_input_ in enumerate(decoder_input_):
                    d_inputs[batchi][di] = d_input_
                    d_masks[batchi][di][:d_input_+1] = 1
                d_outputs[batchi][:len(parsing_index_)] = parsing_index_
                d_output_re[batchi][:len(relation_label_)] = relation_label_
            texts = torch.stack(texts, dim=0)
            assert texts.shape[0] == len(sep_index)
            lengths = pad_sequence(lengths, batch_first=True, padding_value=1)
            speakers = ints_to_tensor(list(speakers))
            turns = ints_to_tensor(list(turns))
            graphs = ints_to_tensor(list(graphs))
            edu_nums = torch.tensor(edu_nums)
            d_inputs = torch.from_numpy(d_inputs).long()
            d_outputs = torch.from_numpy(d_outputs).long()
            d_output_re = torch.from_numpy(d_output_re).long()
            d_masks = torch.from_numpy(d_masks).byte()
            print(d_masks.shape)
            yield texts, sep_index, pairs, graphs, lengths, speakers, turns,\
                  edu_nums,d_inputs,d_outputs,d_output_re,d_masks
    return pool(examples)


def train_collate_fn_new(batch):
    # TODO batch implementation

    output = common_collate_fn(batch)

    # texts, sep_index, pairs, graphs, lengths, speakers, turns, edu_nums, \
    # parsing_index,decoder_input,relation_labels, ids= zip(*examples)
    parsing_index = [instance["parsing_index"] for instance in batch]
    decoder_input = [instance["decoder_input"] for instance in batch]
    relation_labels = [instance["relation_labels"] for instance in batch]
    
    max_edu_seqlen = max([instance["edu_nums"] for instance in batch])
    d_inputs = np.zeros([1, max_edu_seqlen], dtype=np.compat.long)
    d_outputs = np.zeros([1, max_edu_seqlen], dtype=np.compat.long)
    d_output_re = np.zeros([1, max_edu_seqlen], dtype=np.compat.long)
    d_masks = np.zeros([1, max_edu_seqlen, max_edu_seqlen + 1], dtype=np.uint8)
    for di, d_input_ in enumerate(decoder_input[0]):
        d_inputs[0][di] = d_input_
        d_masks[0][di][:d_input_ + 1] = 1
    d_outputs[0][:len(parsing_index[0])] = parsing_index[0]
    d_output_re[0][:len(relation_labels[0])] = relation_labels[0]
    
    # lengths = pad_sequence(lengths, batch_first=True, padding_value=1)
    """texts = torch.stack(texts, dim=0)
    assert texts.shape[0]==len(sep_index)"""
    # speakers = ints_to_tensor(list(speakers))
    # turns = ints_to_tensor(list(turns))
    # graphs = ints_to_tensor(list(graphs))
    # edu_nums = torch.tensor(edu_nums)
    d_inputs = torch.from_numpy(d_inputs).long()
    d_outputs = torch.from_numpy(d_outputs).long()
    d_output_re = torch.from_numpy(d_output_re).long()
    d_masks = torch.from_numpy(d_masks).byte()
    # return texts, sep_index, pairs, graphs, lengths, \
    #        speakers, turns, edu_nums, d_inputs, d_outputs, d_output_re, d_masks,ids

    lengths = pad_sequence([torch.tensor(instance["lengths"]) for instance in batch], batch_first=True, padding_value=1)  # nn.utils.rnn.pad_sequence
    speakers = ints_to_tensor([instance["speakers"] for instance in batch]) if "speakers" in batch[0].keys() else None
    turns = ints_to_tensor([instance["turns"] for instance in batch]) if "turns" in batch[0].keys() else None
    graphs = ints_to_tensor([instance["graphs"] for instance in batch])
    edu_nums = torch.tensor([instance["edu_nums"] for instance in batch])
    pairs = [instance["pairs"] for instance in batch]
    
    output.update(
        {
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
              
              "decoder_input": d_inputs,
            #   "d_outputs": d_outputs,
              "splits_ground": d_outputs,
            #   "d_output_re": d_output_re,
              "nrs_ground": d_output_re,
              "decoder_mask": d_masks,
        }
    )

    return output


def eval_collate_fn(batch):
    # TODO batch implementation

    output = common_collate_fn(batch)

    # texts, sep_index, pairs, graphs, lengths, speakers, turns, edu_nums, \
    # parsing_index,decoder_input,relation_labels, ids= zip(*examples)
    parsing_index = [instance["parsing_index"] for instance in batch]
    decoder_input = [instance["decoder_input"] for instance in batch]
    relation_labels = [instance["relation_labels"] for instance in batch]
    
    max_edu_seqlen = max([instance["edu_nums"] for instance in batch])
    d_inputs = np.zeros([1, max_edu_seqlen], dtype=np.compat.long)
    d_outputs = np.zeros([1, max_edu_seqlen], dtype=np.compat.long)
    d_output_re = np.zeros([1, max_edu_seqlen], dtype=np.compat.long)
    d_masks = np.zeros([1, max_edu_seqlen + 1, max_edu_seqlen + 1], dtype=np.uint8)
    for di, d_input_ in enumerate(decoder_input[0]):
        d_inputs[0][di] = d_input_
        d_masks[0][di][:d_input_ + 1] = 1
    d_outputs[0][:len(parsing_index[0])] = parsing_index[0]
    d_output_re[0][:len(relation_labels[0])] = relation_labels[0]
    
    # lengths = pad_sequence(lengths, batch_first=True, padding_value=1)
    """texts = torch.stack(texts, dim=0)
    assert texts.shape[0]==len(sep_index)"""
    # speakers = ints_to_tensor(list(speakers))
    # turns = ints_to_tensor(list(turns))
    # graphs = ints_to_tensor(list(graphs))
    # edu_nums = torch.tensor(edu_nums)
    d_inputs = torch.from_numpy(d_inputs).long()
    d_outputs = torch.from_numpy(d_outputs).long()
    d_output_re = torch.from_numpy(d_output_re).long()
    d_masks = torch.from_numpy(d_masks).byte()
    # return texts, sep_index, pairs, graphs, lengths, \
    #        speakers, turns, edu_nums, d_inputs, d_outputs, d_output_re, d_masks,ids

    lengths = pad_sequence([torch.tensor(instance["lengths"]) for instance in batch], batch_first=True, padding_value=1)  # nn.utils.rnn.pad_sequence
    speakers = ints_to_tensor([instance["speakers"] for instance in batch]) if "speakers" in batch[0].keys() else None
    turns = ints_to_tensor([instance["turns"] for instance in batch]) if "turns" in batch[0].keys() else None
    graphs = ints_to_tensor([instance["graphs"] for instance in batch])
    edu_nums = torch.tensor([instance["edu_nums"] for instance in batch])
    pairs = [instance["pairs"] for instance in batch]
    
    output.update(
        {
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
              
              "decoder_input": d_inputs,
            #   "d_outputs": d_outputs,
              "splits_ground": d_outputs,
            #   "d_output_re": d_output_re,
              "nrs_ground": d_output_re,
              "decoder_mask": d_masks,
        }
    )

    return output


def trans_structure(predict_parsing_index, prediate_label_index):

    predict_edge_rela = {}
    edge_list = []
    rela_list = []
    assert len(predict_parsing_index)==len(prediate_label_index)
    cur = 1
    for pre_node in predict_parsing_index[1:]:
        pre_node = pre_node - 1
        if pre_node < 0:
            pre_node = 0
        if cur != pre_node:
            edge_list.append((pre_node,cur))
            rela_list.append(prediate_label_index[cur])
        cur += 1
    for index, edge in enumerate(edge_list):
        predict_edge_rela[edge] = rela_list[index]
    return predict_edge_rela

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

def get_node_mask(edu_nums, node_num):
    node_list = []
    for index,edu_num in enumerate(edu_nums):
        temp =  [1]*edu_num+[0]*(node_num-edu_num)
        node_list.append(temp)
    return torch.LongTensor(np.array(node_list,dtype=np.long)).to(edu_nums.device)


def get_mask(node_num, max_edu_dist):
    batch_size, max_num=node_num.size(0), node_num.max()
    mask=torch.arange(max_num).unsqueeze(0).to(node_num.device)<node_num.unsqueeze(1)
    mask=mask.unsqueeze(1).expand(batch_size, max_num, max_num)
    mask=mask&mask.transpose(1,2)
    mask = torch.tril(mask, -1)
    if max_num > max_edu_dist:
        mask = torch.triu(mask, max_edu_dist - max_num)
    return mask


def compute_loss(link_scores, label_scores, graphs, mask, p=False, negative=False):
    #$# print("[*] compute_loss(), called by Model.forward()")
    #$# print(f"mask: {mask}, {mask.shape}")
    link_scores[~mask]=-1e9
    label_mask=(graphs!=0)&mask
    tmp_mask=(graphs.sum(-1)==0)&mask[:,:,0]
    link_mask=label_mask.clone()
    link_mask[:,:,0]=tmp_mask

    #$# print(f"link_scores, after mask: {link_scores}, {link_scores.shape}")
    #$# print(f"link_mask: label_mask = (graphs!=0) & mask -> link_mask[:,:,0] = tmp_mask: {link_mask}, {link_mask.shape}")
    
    link_scores=torch.nn.functional.softmax(link_scores, dim=-1)
    #$# print(f"link_scores_softmax: {link_scores}, {link_scores.shape}")
    #$# print(f"link_scores[link_mask], the final input to loss function: {link_scores[link_mask]}, {link_scores[link_mask].shape}")
    
    link_loss=-torch.log(link_scores[link_mask])
    #$# print(f"link_loss(nllloss): {link_loss}, {link_loss.shape}")

    vocab_size=label_scores.size(-1)
    label_loss=torch.nn.functional.cross_entropy(label_scores[label_mask].reshape(-1, vocab_size), graphs[label_mask].reshape(-1), reduction='none')
    if negative:
        negative_mask=(graphs==0)&mask
        negative_loss=torch.nn.functional.cross_entropy(label_scores[negative_mask].reshape(-1, vocab_size), graphs[negative_mask].reshape(-1),reduction='mean')
        return link_loss, label_loss, negative_loss
    if p:
        return link_loss, label_loss, torch.nn.functional.softmax(label_scores[label_mask],dim=-1)[torch.arange(label_scores[label_mask].size(0)),graphs[mask]]
    return link_loss, label_loss, None  # 12/14



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

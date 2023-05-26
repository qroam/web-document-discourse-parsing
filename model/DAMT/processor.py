from tqdm import tqdm
from typing import Dict, List
from collections import defaultdict
import re
from itertools import product
from operator import itemgetter

import torch
from torch.utils.data import Dataset
import json
import numpy as np

# from ..processor import WebDataProcessor
# from .. import WebDataProcessor
from processor import WebDataProcessor, FATHER_RELATION_dict

# from .utils import *
# FATHER_RELATION_dict = {0: "Title", 1: "Attribute", 2: "Summary", 3: "Elaborate", 4: "Caption", 5: None}


def text_to_id(inputs: List, id_to_text_mapper: Dict):
    if type(inputs) != list:
        inputs = [inputs]
    text_to_id_mapper = {v: k for k, v in id_to_text_mapper.items()}
    return [text_to_id_mapper[text] if text in text_to_id_mapper.keys() else text_to_id_mapper[None] for text in inputs]


class DiscourseGraph:
    def __init__(self, dialogue, pairs):
        self.dialogue = dialogue
        self.pairs = pairs
        # self.edu_num = len(self.dialogue['edus'])
        self.edu_num = len(self.dialogue["Content"])
        self.paths = self.get_graph(pairs, self.edu_num)
        self.speaker_paths = self.get_speaker_paths(dialogue)  # None
        self.turn_paths = self.get_turn_paths(dialogue)  # None

    @staticmethod
    def print_path(path):
        for row in path:
            print([col for col in row])

    @staticmethod
    def get_speaker_paths(dialogue):  # Our data do not have speaker feature
        return
        """speaker_size = len(dialogue['edus']) + 1
        speaker_4edu = ['None']
        for edu in dialogue['edus']:
            if isinstance(edu['speaker'], str):
                speaker_4edu.append(edu['speaker'])
            else:
                speaker_4edu.append('None')
        speaker_4edu = np.array(speaker_4edu)
        speaker_4edu_Aside = speaker_4edu.repeat(speaker_size).reshape(speaker_size, speaker_size)
        speaker_4edu_Bside = speaker_4edu_Aside.transpose()
        return (speaker_4edu_Aside == speaker_4edu_Bside).astype(np.long)"""

    @staticmethod
    def get_turn_paths(dialogue):  # Our data do not have turn feature
        return
        """turn_size = len(dialogue['edus']) + 1
        turns = [0] + [edu['turn'] for edu in dialogue['edus']]
        turns = np.array(turns)
        turn_Aside = turns.repeat(turn_size).reshape(turn_size, turn_size)
        turn_Bside = turn_Aside.transpose()
        return (turn_Aside == turn_Bside).astype(np.long)"""

    @staticmethod
    def get_coreference_path(dialogue):  # No usage
        coreferences = []
        edu_num = len(dialogue['edus'])
        path = np.zeros((edu_num + 1, edu_num + 1), dtype=np.long)
        if 'solu' in dialogue:
            for cluster in dialogue['solu']:
                coreferences.append([k for (k, v) in cluster])
            for cluster in coreferences:
                for (x, y) in list(product(cluster, cluster)):
                    if x != y:
                        x, y = x + 1, y + 1
                        path[x][y] = 1
        return path.tolist()

    @staticmethod
    def get_graph(pairs, edu_num):
        node_num = edu_num + 1
        graph = np.zeros([node_num, node_num], dtype=np.long)
        for (x, y), label in pairs.items():
            graph[y + 1][x + 1] = label  # TODO 1107, 1->2, 2->3, ..., 5->0
        return graph.tolist()


class DAMTProcessor(WebDataProcessor):
    # That is just similar with SSAProcessor
    def __init__(self, args, tokenizer):
        super(DAMTProcessor, self).__init__(args, tokenizer)
        # self.text2id_mappers
        self.parent_relation_mapping = args.parent_relation_mapping

    
    def _get_discourse_graph(self, instance):
        # called by main.py
        def get_parsing_index_and_labels(pairs,edu_num):
            new_pairs ={}
            for a ,b in pairs.items():
                new_pairs[(a[0]+1,a[1]+1)] = b
            parsing_index = [0]
            labels = [0]
            for i in range(2,edu_num+1):
                find_pre_edg  = False
                for a,b in new_pairs.items():
                    if a[1]==i and a[0]<a[1]:
                        parsing_index.append(a[0])
                        labels.append(b)
                        find_pre_edg = True
                        break
                if not find_pre_edg:
                    parsing_index.append(i)
                    labels.append(0)
            return torch.tensor(parsing_index), torch.tensor(labels)

        def get_decode_index(edu_num):
            return torch.tensor([index+1 for index in range(edu_num)])

        """for dialogue in self.dialogues:
            edu_nums = len(dialogue['edus'])
            pairs = {(relation['x'], relation['y']): self.type2ids[relation['type']]
                     for relation in dialogue['relations']}
            discourse_graph = DiscourseGraph(dialogue, pairs)
            dialogue['graph'] = discourse_graph
            dialogue['parsing_index'],dialogue['labels']=get_parsing_index_and_labels(pairs,edu_nums)
            dialogue['decoder_input'] =get_decode_index(edu_nums)"""
        edu_nums = len(instance["Content"])
        pairs = {(y, x): instance["Father_Relation_ids"][x] for x, y in enumerate(instance["Father"])}
        """pairs = {k: v + 1 if self.parent_relation_mapping[v] is not None else 0 for k, v in pairs.items()}"""
        pairs = {k: v + 1 for k, v in pairs.items()}  # 12/14, relation "NA" is 0, others in sequential 1, 2, ...
        # (child node id: parent node id)
        graph = DiscourseGraph(instance, pairs)
        parsing_index, labels = get_parsing_index_and_labels(pairs, edu_nums)
        decoder_input = get_decode_index(edu_nums)
        return graph, parsing_index, labels, decoder_input
        
    
    # @staticmethod
    def add_features(self, instance,):
        """
        def forward(self, inputs:Dict, training_tuple_ids):
        where training_tuple_ids be like: [((0,1,2,3),1), ((0,1,3),0), ...]
        :param instance:
        :return:
        """
        lengths = [len(node) for node in instance["Content"]]
        edu_nums = len(instance["Content"])
        graph, parsing_index, labels, decoder_input = self._get_discourse_graph(instance)
        instance.update(
            {
                "lengths": lengths,  # List[int], (num_edu), indicating the number of tokens in each EDU in the instance
                "edu_nums": edu_nums,  # # int, the number of EDUs in this instance
                # "graphs": graph,
                "graphs": graph.paths,  # 
                "pairs": graph.pairs,  # 
                "speakers": None,  # our data have no this information
                "turns": None,  # our data have no this information

                "parsing_index": parsing_index,  # dialogue['parsing_index']，torch.tensor，father ids
                "decoder_input": decoder_input,  # dialogue['decoder_input']，torch.tensor，e.g. [1,2,3,4,5]
                "relation_labels": labels,  # dialogue['labels']，torch.tensor，father id labels
            }
        )
        return instance
    

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
        self.speaker_paths = self.get_speaker_paths(dialogue)
        self.turn_paths = self.get_turn_paths(dialogue)

    @staticmethod
    def print_path(path):
        for row in path:
            print([col for col in row])

    @staticmethod
    def get_speaker_paths(dialogue):
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
    def get_turn_paths(dialogue):
        return
        """turn_size = len(dialogue['edus']) + 1
        turns = [0] + [edu['turn'] for edu in dialogue['edus']]
        turns = np.array(turns)
        turn_Aside = turns.repeat(turn_size).reshape(turn_size, turn_size)
        turn_Bside = turn_Aside.transpose()
        return (turn_Aside == turn_Bside).astype(np.long)"""

    @staticmethod
    def get_coreference_path(dialogue):
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
        # pairs: (father_id, child_id): relation_label
        node_num = edu_num + 1  # + dummy node
        graph = np.zeros([node_num, node_num], dtype=np.long)
        for (x, y), label in pairs.items():
            graph[y + 1][x + 1] = label  # TODO 1107, 1->2, 2->3, ..., 5->0
        return graph.tolist()


class SSAProcessor(WebDataProcessor):
    def __init__(self, args, tokenizer):
        super(SSAProcessor, self).__init__(args, tokenizer)
        # self.text2id_mappers
        self.parent_relation_mapping = args.parent_relation_mapping

    # @staticmethod
    def _get_discourse_graph(self, instance):
        # pairs = {(x, y): instance["Father_Relation_ids"][x] for x, y in enumerate(instance["Father"])}
        pairs = {(y, x): instance["Father_Relation_ids"][x] for x, y in enumerate(instance["Father"])}  # (father_id, child_id): relation_label
        """pairs = {k: v + 1 if self.parent_relation_mapping[v] is not None else 0 for k, v in pairs.items()}"""
        pairs = {k: v + 1 for k, v in pairs.items()}  # 12/14, relation "NA" is 0, others in sequential 1, 2, ...
        # (child node id: parent node id)
        return DiscourseGraph(instance, pairs)
        """for dialogue in self.dialogues:
            pairs = {(relation['x'], relation['y']): self.type2ids[relation['type']]
                     for relation in dialogue['relations']}
            discourse_graph = DiscourseGraph(dialogue, pairs)
            dialogue['graph'] = discourse_graph"""

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
        graph = self._get_discourse_graph(instance)
        instance.update(
            {
                "lengths": lengths,
                "edu_nums": edu_nums,
                # "graphs": graph,
                "graphs": graph.paths,
                "pairs": graph.pairs,
                "speakers": None,
                "turns": None,
            }
        )
        return instance


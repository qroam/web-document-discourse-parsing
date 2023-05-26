from tqdm import tqdm
from typing import Dict, List
from collections import defaultdict

# from ..processor import WebDataProcessor
# from .. import WebDataProcessor
from processor import WebDataProcessor, FATHER_RELATION_dict


# FATHER_RELATION_dict = {0: "Title", 1: "Attribute", 2: "Summary", 3: "Elaborate", 4: "Caption", 5: None}


"""def text_to_id(inputs: List, id_to_text_mapper: Dict):
    if type(inputs) != list:
        inputs = [inputs]
    text_to_id_mapper = {v: k for k, v in id_to_text_mapper.items()}
    return [text_to_id_mapper[text] if text in text_to_id_mapper.keys() else text_to_id_mapper[None] for text in inputs]"""

def text_to_id(inputs: List, id_to_text_mapper: Dict):
    # 12/13 UPDATE
    if type(inputs) != list:
        inputs = [inputs]
    text_to_id_mapper = {v: k for k, v in id_to_text_mapper.items()}
    for label in inputs:
        assert label in list(text_to_id_mapper.keys()) + ["False"]
    label_ids = [text_to_id_mapper[text] if text in text_to_id_mapper.keys() else len(FATHER_RELATION_dict.keys()) for text in inputs]
    label_ids = [i if i != -1 else len(FATHER_RELATION_dict.keys())-1 for i in label_ids]  # "NA" labels
    return label_ids


def _rightmost_branch(idx, parent_to_child):
    rightmost_branch = []
    rightmost_branch.append(idx)
    while (parent_to_child[idx] != []):
        idx = parent_to_child[idx][-1]
        rightmost_branch.append(idx)
    return rightmost_branch


def get_rightmost_branch_position(parent_to_child):
    # dummy node id -> -1
    return _rightmost_branch(-1, parent_to_child)


def get_rightmost_branch_plus_title_position(parent_to_child):
    # dummy node id -> -1
    return list(set(_rightmost_branch(-1, parent_to_child) + _rightmost_branch(0, parent_to_child)))


def get_all_position(parent_to_child):
    # dummy node id -> -1
    return list(dict(parent_to_child).keys())


def get_context_and_label_of_insert_position(father_index, current_node_index, parent_to_child, paragraphs, father_ids,
                                             father_relations):
    father = father_index
    siblings = parent_to_child[father_index]
    context_node_ids = [father] + siblings + [current_node_index]
    context_paragraphs = [paragraphs[i] for i in context_node_ids]
    if father_ids[current_node_index] != father_index:
        label = "False"
    else:
        label = father_relations[current_node_index]
    return context_node_ids, context_paragraphs, label

def get_context_and_label_of_insert_position_new(father_index, current_node_index, parent_to_child, paragraphs, father_ids,
                                             father_relations):
    father = father_index
    siblings = parent_to_child[father_index]
    if father == -1:  # 11/28
        context_node_ids = [father] + [current_node_index]
    else:
        context_node_ids = [father] + siblings + [current_node_index]
    context_paragraphs = [paragraphs[i] for i in context_node_ids]
    if father_ids[current_node_index] != father_index:
        label = "False"
    else:
        label = father_relations[current_node_index]
    return context_node_ids, context_paragraphs, label

def get_context_of_insert_position(father_index, current_node_index, parent_to_child):  #, paragraphs,):
    father = father_index
    siblings = parent_to_child[father_index]
    context_node_ids = [father] + siblings + [current_node_index]
    # context_paragraphs = [paragraphs[i] for i in context_node_ids]
    return context_node_ids#, context_paragraphs

def get_context_of_insert_position_new(father_index, current_node_index, parent_to_child):  #, paragraphs,):
    # modified 11/28, for insertion position under dummy node, its context_paragraphs 
    father = father_index
    if father == -1:  # 11/28
        context_node_ids = [father] + [current_node_index]
    else:
        siblings = parent_to_child[father_index]
        context_node_ids = [father] + siblings + [current_node_index]
    # context_paragraphs = [paragraphs[i] for i in context_node_ids]
    return context_node_ids#, context_paragraphs


def get_parent_to_children_dict(father_ids):
    parent_to_child = defaultdict(list)
    parent_to_child[-1] = []
    for i, father_i in enumerate(father_ids):
        parent_to_child[father_i].append(i)
        parent_to_child[i] = []
    return dict(parent_to_child)


position_method = {
    "right": get_rightmost_branch_position,
    "right+maintitle": get_rightmost_branch_plus_title_position,
    "all": get_all_position,
}


class PutOrSkipProcessor(WebDataProcessor):
    def __init__(self, args, tokenizer):
        super(PutOrSkipProcessor, self).__init__(args, tokenizer)
        # self.text2id_mappers
        self.train_possible_position = args.train_possible_position
        self.train_position_method = position_method[args.train_possible_position]

    @staticmethod
    def _get_training_tuples(instance, possible_position_mathod):
        training_tuples = []
        paragraphs = instance["Content"]
        father_ids = instance["Father"]
        father_relations = instance["Father_Relation"]
        previous_ids = instance["Previous"]
        parent_to_child = defaultdict(list)
        parent_to_child[-1] = []
        for i, paragraph in enumerate(paragraphs):
            # keep track of the rightmost branch of current tree

            # get all possible insertion positions on current tree for current added node
            # possible_insertion_positions = get_all_position(parent_to_child)
            possible_insertion_positions = possible_position_mathod(parent_to_child)
            for position in possible_insertion_positions:
                training_tuple = get_context_and_label_of_insert_position_new(
                    father_index=position,
                    current_node_index=i,
                    parent_to_child=parent_to_child,
                    paragraphs=paragraphs,
                    father_ids=father_ids,
                    father_relations=father_relations)
                training_tuples.append(training_tuple)

            # update current added node to the tree
            father_i = father_ids[i]
            parent_to_child[father_i].append(i)
            parent_to_child[i] = []
        return training_tuples

    # @staticmethod
    def add_features(self, instance,):
        """
        def forward(self, inputs:Dict, training_tuple_ids):
        where training_tuple_ids be like: [((0,1,2,3),1), ((0,1,3),0), ...]
        :param instance:
        :return:
        """
        training_tuples = PutOrSkipProcessor._get_training_tuples(instance, self.train_position_method)
        context_node_ids = [tuple(t[0]) for t in training_tuples]
        context_label = [t[2] for t in training_tuples]
        context_label_ids = text_to_id(context_label, FATHER_RELATION_dict)
        assert len(context_label_ids) == len(context_node_ids)
        instance["training_tuple_ids"] = [(a, b) for a, b in zip(context_node_ids, context_label_ids)]
        return instance


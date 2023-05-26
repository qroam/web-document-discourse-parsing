from typing import List
from collections import defaultdict

import os
import argparse
import torch
import random
import time
import numpy as np
from transformers.optimization import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, get_constant_schedule, get_cosine_schedule_with_warmup
from torch.optim import Adam, SGD
from transformers.optimization import AdamW
from processor import PREVIOUS_RELATION_dict, FATHER_RELATION_dict


class DocumentRecorder:
    """
    we do not expect to initialize this object in main functions, rather, we provide a from_data() method to cover the
    corresponding logics, which makes the main functions only need to deal with the data stream of each batch, without
    concerning details of how they are recorded
    """
    def __init__(self, golden_parent_ids, golden_parent_relations, golden_previous_ids, golden_previous_relations,
                 predicted_parent_ids=None, predicted_parent_relations=None,
                 predicted_previous_ids=None, predicted_previous_relations=None,
                 id=-1, node_modal=None):
        """batch_golden_parent_ids, batch_golden_previous_ids, batch_golden_parent_relations, batch_golden_previous_relations = [], [], [], []
        batch_pred_parent_ids, batch_pred_previous_ids, batch_pred_parent_relations, batch_pred_previous_relations = [], [], [], []
        length_of_documents = []
        node_modals = []"""
        self.golden_parent_ids = golden_parent_ids or []
        self.golden_parent_relations = golden_parent_relations or []
        self.golden_previous_ids = golden_previous_ids or []
        self.golden_previous_relations = golden_previous_relations or []
        # print(predicted_parent_ids)
        self.predicted_parent_ids = predicted_parent_ids or []
        self.predicted_parent_relations = predicted_parent_relations or []
        self.predicted_previous_ids = predicted_previous_ids or []
        self.predicted_previous_relations = predicted_previous_relations or []
        
        self.predicted_parent_relations = [FATHER_RELATION_dict[i] for i in self.predicted_parent_relations]
        self.predicted_previous_relations = [PREVIOUS_RELATION_dict[i] for i in self.predicted_previous_relations]

        # If one node has the dommy root node to be its parent node, their parent-child relation ship shall be "NA" by default
        # Same as previous relations
        # "NA", rather thah None
        self.predicted_parent_relations = [i if self.predicted_parent_ids[idx] != -1 else "NA" for idx, i in enumerate(self.predicted_parent_relations)]
        self.predicted_previous_relations = [i if self.predicted_previous_ids[idx] != -1 else "NA" for idx, i in enumerate(self.predicted_previous_relations)]
        
        self.id = id
        self.node_modal = node_modal or []
        # print([item for item in vars(self).items()])  # for debug

    def __repr__(self):
        """
        for case study
        :return: str
        """
        # output_file.write(f"golden parent ids: {case_golden_parent_ids}" + "\n")
        # output_file.write(f"predict parent ids: {case_predict_parent_ids}" + "\n")
        # output_file.write(f"{difference_between_list(case_golden_parent_ids, case_predict_parent_ids)}" + "\n")
        # output_file.write(f"golden parent relations: {case_golden_parent_relations}" + "\n")
        # output_file.write(f"predict parent relations: {case_predict_parent_relations}" + "\n")
        # output_file.write(f"golden previous ids: {case_golden_previous_ids}" + "\n")
        # output_file.write(f"predict previous ids: {case_predict_previous_ids}" + "\n")
        # output_file.write(f"golden previous relations: {case_golden_previous_relations}" + "\n")
        # output_file.write(f"predict previous relations: {case_predict_previous_relations}" + "\n")
        self_string = f"id = {self.id}" + "\n" + \
                      f"golden parent ids: {self.golden_parent_ids}" + "\n" + \
                      f"predict parent ids: {self.predicted_parent_ids}" + "\n" + \
                      f"{difference_between_list(self.golden_parent_ids, self.predicted_parent_ids)}" + "\n" + \
                      f"golden parent relations: {self.golden_parent_relations}" + "\n" + \
                      f"predict parent relations: {self.predicted_parent_relations}" + "\n" + \
                      f"{difference_between_list(self.golden_parent_relations, self.predicted_parent_relations)}" + "\n" + \
                      f"golden previous ids: {self.golden_previous_ids}" + "\n" + \
                      f"predict previous ids: {self.predicted_previous_ids}" + "\n" + \
                      f"{difference_between_list(self.golden_previous_ids, self.predicted_previous_ids)}" + "\n" + \
                      f"golden previous relations: {self.golden_previous_relations}" + "\n" + \
                      f"predict previous relations: {self.predicted_previous_relations}" + "\n" + \
                      f"{difference_between_list(self.golden_previous_relations, self.predicted_previous_relations)}" + "\n"
        return self_string

    @staticmethod
    def from_data(batch_meta, outputs, model_implemented_funtions):
        """
        both batch_meta and outputs are batch-level variavles, this method returns a List of Documents, each of them
        represents the golden and pred as well as meta information of an instance in the batch
        :param batch_meta:
        :param outputs:
        :param model_implemented_funtions:
        :return:
        """
        list_of_document = []

        ids = batch_meta["ids"]
        for i, document_id in enumerate(ids):
            node_modal, golden_parent_ids, golden_parent_relations, golden_previous_ids, golden_previous_relations = \
                batch_meta["node_modal"][i], batch_meta["golden_parent_ids"][i], batch_meta["golden_parent_relations"][i],\
                batch_meta["golden_previous_ids"][i], batch_meta["golden_previous_relations"][i]

            """model_implemented_funtions.update({
                "parent_ids": outputs["father_ids"] is not None,
                "parent_relations": outputs["father_labels"] is not None,
                "previous_ids": outputs["previous_ids"] is not None,
                "previous_relations": outputs["previous_labels"] is not None,
            })"""
            predicted_parent_ids = None
            predicted_parent_relations = None
            predicted_previous_ids = None
            predicted_previous_relations = None
            if model_implemented_funtions["parent_ids"]:
                predicted_parent_ids = outputs['father_ids'][i].tolist() if type(outputs['father_ids'])==torch.Tensor else outputs['father_ids'][i]
            if model_implemented_funtions["parent_relations"]:
                predicted_parent_relations = outputs['father_labels'][i].tolist() if type(outputs['father_labels'])==torch.Tensor else outputs['father_labels'][i]
            if model_implemented_funtions["previous_ids"]:
                predicted_previous_ids = outputs['previous_ids'][i].tolist() if type(outputs['previous_ids'])==torch.Tensor else outputs['previous_ids'][i]
            if model_implemented_funtions["previous_relations"]:
                predicted_previous_relations = outputs['previous_labels'][i].tolist() if type(outputs['previous_labels'])==torch.Tensor else outputs['previous_labels'][i]
            document_case = DocumentRecorder(
                golden_parent_ids, golden_parent_relations, golden_previous_ids, golden_previous_relations,
                predicted_parent_ids, predicted_parent_relations, predicted_previous_ids, predicted_previous_relations,
                document_id, node_modal
            )
            list_of_document.append(document_case)

        """node_modals += flatten_list(batch_meta["node_modal"])

        batch_golden_parent_ids += flatten_list(batch_meta["golden_parent_ids"])
        batch_golden_previous_ids += flatten_list(batch_meta["golden_previous_ids"])
        batch_golden_parent_relations += flatten_list(batch_meta["golden_parent_relations"])
        batch_golden_previous_relations += flatten_list(batch_meta["golden_previous_relations"])

        length_of_documents += [len(document) for document in batch_meta["golden_parent_ids"]]

        predict_parent_ids = outputs["father_ids"].reshape(-1).tolist() if type(
            outputs["father_ids"]) == torch.tensor else outputs["father_ids"]
        predict_parent_labels = outputs["father_labels"].reshape(-1).tolist() if type(
            outputs["father_labels"]) == torch.tensor else outputs["father_labels"]
        predict_previous_ids = outputs["previous_ids"].reshape(-1).tolist() if type(
            outputs["previous_ids"]) == torch.tensor else outputs["previous_ids"]
        predict_previous_relations = outputs["previous_labels"].reshape(-1).tolist() if type(
            outputs["previous_labels"]) == torch.tensor else outputs["previous_labels"]
        
        batch_pred_parent_ids += predict_parent_ids
        batch_pred_previous_ids += predict_previous_ids
        batch_pred_parent_relations += predict_parent_labels
        # batch_pred_previous_relations += predict_previous_relations
        batch_pred_previous_relations += predict_previous_relations
        
        for i, document_id in enumerate(ids):
            output_file.write(f"Evaluate on {tag} set" + "\t" + f"id = {document_id}" + "\n")
            case_golden_parent_ids = batch_meta['golden_parent_ids'][i]
            case_predict_parent_ids = outputs['father_ids'][i].tolist()
            case_golden_parent_relations = batch_meta['golden_parent_relations'][i]
            case_predict_parent_relations = outputs['father_labels'][i].tolist()
            case_golden_previous_ids = batch_meta['golden_previous_ids'][i]
            case_predict_previous_ids = outputs['previous_ids'][i].tolist()
            case_golden_previous_relations = batch_meta['golden_previous_relations'][i]
            case_predict_previous_relations = outputs['previous_labels'][i].tolist()
            
            output_file.write(f"golden parent ids: {case_golden_parent_ids}" + "\n")
            output_file.write(f"predict parent ids: {case_predict_parent_ids}" + "\n")
            output_file.write(f"{difference_between_list(case_golden_parent_ids, case_predict_parent_ids)}" + "\n")
            output_file.write(f"golden parent relations: {case_golden_parent_relations}" + "\n")
            output_file.write(f"predict parent relations: {case_predict_parent_relations}" + "\n")
            output_file.write(f"golden previous ids: {case_golden_previous_ids}" + "\n")
            output_file.write(f"predict previous ids: {case_predict_previous_ids}" + "\n")
            output_file.write(f"golden previous relations: {case_golden_previous_relations}" + "\n")
            output_file.write(f"predict previous relations: {case_predict_previous_relations}" + "\n")
        """

        return list_of_document


def save_devide(a, b):
    if b != 0:
        return a / b
    else:
        return 0.0


"""def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)"""

def seed_everything(seed=256):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def set_seed(args):
    if args.seed >= 0:
        seed = args.seed
    else:
        seed = random.randint(10000, 99999)
        args.seed = seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(int(seed / 10))
    np.random.seed(seed)
    # torch.manual_seed(int(seed / 100))
    torch.manual_seed(seed)
    if torch.cuda.is_available() and args.device_id >= 0:
        torch.cuda.set_device(args.device_id)
        # torch.cuda.manual_seed(int(seed / 1000))
        # torch.cuda.manual_seed_all(int(seed / 1000))
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def prepare_all_argparsers(train_env_map):
    parser = argparse.ArgumentParser()

    sub_parsers = parser.add_subparsers(dest='Model Name')
    sub_parsers.required = True

    for key, value in train_env_map.items():
        sub_parser = sub_parsers.add_parser(key)
        sub_parser.set_defaults(model_name=key)

        add_common_arguments(sub_parser)
        value.add_arguments(sub_parser)

    return parser

def flatten_list(lst):
    out = []
    for item in lst:
        if type(item) == list:
            flatten_item = flatten_list(item)
            out += flatten_item
        else:
            out.append(item)
    return out

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



def add_common_arguments(parser):
    # data dir

    parser.add_argument("--train_set_dir", default="data/train", type=str)
    parser.add_argument("--dev_set_dir", default="data/dev", type=str)
    parser.add_argument("--test_set_dir", default="data/test", type=str)

    parser.add_argument("--data_cache_path", default="data", type=str)

    parser.add_argument("--data_cache_key", default="dafault", type=str)
    
    # checkpoint name, only used when test
    # parser.add_argument("--test_checkpoint_id", default=0, type=int, )
    parser.add_argument("--test_checkpoint_dir", default="", type=str, )

    # log dir
    parser.add_argument("--log_dir", default="log.txt", type=str, )

    # data preprocessing settings
    # parser.add_argument("--combine_before", default=False, action="store_true",
    #                     help="doing combine in the data preprocessing stage.")
    parser.add_argument("--combine_before", default=True,
                        help="doing combine in the data preprocessing stage.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                            "than this will be truncated.")
    parser.add_argument("--preprocess_figure_node", default=False, action="store_true",
                    help="whether to convert url of all figures to a unified linguistic utterance, e.g. 'tu-pian', during data processing")
    parser.add_argument("--discard_figure_node", default=False, action="store_true",
                    help="whether to discard all figure nodes in document data")
    parser.add_argument("--max_paragraph_num", default=200, type=int, )  # TODO: this argument seems not play a role in the code

    parser.add_argument("--load_cached_features", default=False, action="store_true")


    # common settings
    parser.add_argument("--test_only", default=False, action="store_true",
                        help="Directly doing test on test set, a pretrained model checkpoint should be given")
    parser.add_argument("--device_id", default=0, type=int, help="device id for GPU training.")
    parser.add_argument("--save_checkpoint", default=False, type=bool, help="save checkpoint for each epoch.")
    # parser.add_argument("--golden_parent_when_evaluate", default=False, action="store_true")

    
    # model checkpoint saving dir
    parser.add_argument("--model_path", default="model_checkpoint", type=str)

    
    parser.add_argument('--wordvec_path', default=None, type=str,
                       help="word vector file path, if using")

    # transformers settings
    # parser.add_argument("--config_name", default="bert-base-chinese", type=str,
    #                     help="Pretrained config name or path if not the same as model_name")
    # parser.add_argument("--tokenizer_name", default="bert-base-chinese", type=str,
    #                     help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--config_name", default=None, type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default=None, type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--model_name_or_path", default="bert-base-chinese", type=str)
    # bert-base-chinese
    # hfl/chinese-roberta-wwm-ext-large
    parser.add_argument("--text_encoder_type", default="local", type=str, choices=["local", "pair", "concate"])
    parser.add_argument("--max_seq_length_for_concate", default=1024, type=int, )
    
    
    # label settings
    parser.add_argument("--use_parent_relation_fine", default=False, action="store_true")
    parser.add_argument("--use_previous_relation_fine", default=False, action="store_true")

    # other module settings
    parser.add_argument("--use_global_encoder", default=False, action="store_true")
    parser.add_argument("--global_encoder_type", default="gru", type=str, choices=["rnn","lstm","gru", "transformer"])
    
    parser.add_argument("--use_html_embedding", default=False, action="store_true")
    parser.add_argument("--html_embedding_dim", default=64, type=int,)
    # parser.add_argument("--html_vocab_dir", default="data125_checked/html_vocab.txt", type=str)
    # parser.add_argument("--html_vocab_dir", default="data224/html_vocab.txt", type=str)
    parser.add_argument("--html_vocab_dir", default="data/html_vocab.txt", type=str)

    parser.add_argument("--use_xpath_embedding", default=False, action="store_true")    
    parser.add_argument("--xpath_node_embedding_dim", default=64, type=int,)
    parser.add_argument("--xpath_index_embedding_dim", default=64, type=int,)
    parser.add_argument("--xpath_encoder_type", default="ffnn", type=str, choices=["rnn","ffnn"])
    # parser.add_argument("--xpath_vocab_dir", default="data125_checked/xpath_vocab.txt", type=str)
    # parser.add_argument("--xpath_vocab_dir", default="data224/xpath_vocab.txt", type=str)
    parser.add_argument("--xpath_vocab_dir", default="data/xpath_vocab.txt", type=str)

    parser.add_argument("--max_xpath_index_value", default=101, type=int)

    parser.add_argument("--use_position_embedding", default=False, action="store_true")  # TODO not used
    parser.add_argument("--position_embedding_dim", default=64, type=int,)  # TODO not used

    parser.add_argument("--use_relative_position_embedding", default=False, action="store_true")
    parser.add_argument("--relative_position_embedding_dim", default=64, type=int,)


    parser.add_argument("--modality_aggregation_method", default="concat-project", type=str,
        choices=["concat", "concat-project", "add", "linear-combination-single-weight", "linear-combination-complex-weight"])

    parser.add_argument("--no_text_information", default=False, action="store_true")

    # parser.add_argument("--pair_interation_method", default="bilinear", choices=["bilinear", "biaffine", "concate-mlp", "concate-linear"])
    parser.add_argument("--pair_interation_method_parent", default="variable-class-biaffine", choices=["bilinear", "biaffine", "variable-class-biaffine", "concate-mlp", "concate-linear"])
    parser.add_argument("--pair_interation_method_parent_label", default="biaffine", choices=["bilinear", "biaffine", "concate-mlp", "concate-linear"])
    parser.add_argument("--pair_interation_method_previous_label", default="biaffine", choices=["bilinear", "biaffine", "concate-mlp", "concate-linear"])


    # task setting
    parser.add_argument("--use_previous_joint_loss", default=False, action="store_true",
                        help="Add a pairclassifier(default) to the modle and use the previous task classification loss to jointly train the model")
    parser.add_argument("--previous_loss_ratio", default=0.3, type=float)

    # training & optimization setting
    parser.add_argument("--num_train_epochs", default=100, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--train_batch_size", default=1, type=int,
                    help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=1, type=int,
                        help="Batch size for testing.")

    parser.add_argument("--transformer_learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam on finetuning PTLM.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam on other modules.")
    parser.add_argument("--gradient_accumulation_steps", default=8, type=int,
                        help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--optimizer", default="AdamW", type=str, choices=["SGD", "Adam", "AdamW"],
                        help="type of optimizer.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--earlystop", default=-1, type=int)

    parser.add_argument("--scheduler", default="linear", type=str, choices=["constant", "warmup", "linear", "cosine"],
                        help="type of learning rate scheduler.")
    
    parser.add_argument("--seed", type=int, default=42,
                    help="random seed for initialization")


def prepare_optimizer(args, param_groups):
    if args.optimizer == "SGD":
        optimizer = SGD(param_groups, lr=args.learning_rate)
    elif args.optimizer == "Adam":
        optimizer = Adam(param_groups, lr=args.learning_rate, eps=args.adam_epsilon)
    elif args.optimizer == "AdamW":
    # else:
        optimizer = AdamW(param_groups, lr=args.learning_rate, eps=args.adam_epsilon)
    return optimizer

def prepare_scheduler(name, optimizer, num_warmup_steps, num_training_steps):
    scheduler_dict = {
        "linear": get_linear_schedule_with_warmup,
        "warmup": get_constant_schedule_with_warmup,
        "constant": get_constant_schedule,
        "cosine": get_cosine_schedule_with_warmup
    }
    param_dict = {"optimizer":optimizer, "num_warmup_steps":num_warmup_steps, "num_training_steps":num_training_steps}
    if name == "warmup":
        param_dict.pop("num_training_steps")
    if name == "constant":
        param_dict.pop("num_training_steps")
        param_dict.pop("num_warmup_steps")
    scheduler = scheduler_dict[name](**param_dict)
    return scheduler

def get_acc(goldens, preds, mask_id=-1):
    assert len(goldens) == len(preds)
    correct = 0
    total = 0
    for g, p in zip(goldens, preds):
        if g == mask_id:
            continue
        if g == p:
            correct += 1
        total += 1
    return correct / total * 100


def difference_between_list(golden, predict):
    if not predict:
        return {}
    assert len(golden) == len(predict), (golden, predict)
    error_dict = {}
    for i, (g, p) in enumerate(zip(golden, predict)):
        if g != p:
            error_dict[i] = f"{g} -> {p}"
    # print(error_dict)
    return error_dict


def create_log(args):
    with open(args.log_dir, "a+", encoding="utf-8") as metric_file:
        metric_file.write("\n" + "="*20 + "\n")
        metric_file.write(f"train_batch_size = {args.train_batch_size} \
        gradient_accumulation_steps = {args.gradient_accumulation_steps} \
        learning_rate = {args.learning_rate} \
        loss_type = {args.loss_type} \
        alpha = {args.alpha}" + "\n")

def get_localtime():
    struct_time = time.localtime()
    year = struct_time[0]
    month = struct_time[1]
    day = struct_time[2]
    hour = struct_time[3]
    minute = struct_time[4]
    second = struct_time[5]
    time_string = f"{year}-{month}-{day}_{hour}:{minute}:{second}"
    return time_string

def get_time_lag(t0):
    t1 = time.time()
    return f"{(t1 - t0)//3600} hours, {((t1 - t0) - ((t1 - t0)//3600)*3600)//60} mins, {(t1 - t0) % 60} seconds"

def predict_previous_relations(outputs, classifier, encodings, predicted_father_ids=None, golden_previous_ids=None, golden_previous_labels=None, loss_ratio=0.3):
    """
    :param classifier: a specific nn.Module, PairClassifier by default
    :param encodings: (batch_size, max_num_node, hidden_dim)
    :param father_node_logit_scores: (batch_size, max_num_node, max_num_node)
    :param golden_previous_ids: (batch_size, max_num_node,)
    :param golden_previous_labels: (batch_size, max_num_node, int:2or3)
    """

    if golden_previous_ids is None:  # inference
        # fathers = torch.argmax(father_node_logit_scores, dim=2)
        fathers = predicted_father_ids.tolist() if type(predicted_father_ids) == torch.Tensor else predicted_father_ids
        ###### previous_id_list = father_id_to_previous_id(fathers)
        previous_id_list =  father_id_to_previous_id([[idx+1 for idx in father_ids] for father_ids in fathers])
        previous_id_list = torch.tensor(previous_id_list).to(encodings.device)

        # add previous_ids to the outputs at inference time
        if "previous_ids" not in outputs.keys() or outputs["previous_ids"] is None:
            ###### predicted_previous = father_id_to_previous_id(predicted_links[:, 1:].detach().tolist())
            # predicted_previous = predicted_previous[:, 1:]
            ###### predicted_previous = torch.tensor(predicted_previous)[:, 1:] - 1
            outputs["previous_ids"] = (previous_id_list[:, 1:] - 1).detach()
    
    else: # train, teacher forcing
        previous_id_list = golden_previous_ids

    (previous_relations_logit_scores, previous_loss) = classifier(encodings, previous_id_list, golden_previous_labels)
    # print(previous_relations_logit_scores)
    # print(previous_loss)
    # if previous_loss is not None:
    #     return {
    #         "previous_loss": previous_loss,
    #         "previous_labels": torch.argmax(previous_relations_logit_scores, dim=-1)
    #     }
    # else:
    #     return {"previous_labels": torch.argmax(previous_relations_logit_scores, dim=-1)}
    if previous_loss is not None and "loss" in outputs.keys():
        outputs["original_loss"] = outputs["loss"].clone()
        outputs["loss"] = loss_ratio * previous_loss + (1-loss_ratio) * outputs["loss"]
        # outputs.update(previous_relation_outputs)
        outputs["previous_loss"] = previous_loss
    outputs["previous_labels"] = torch.argmax(previous_relations_logit_scores, dim=-1)
    # return {
    #         "previous_loss": previous_loss,
    #         "previous_labels": torch.argmax(previous_relations_logit_scores, dim=-1)
    #     }
    return outputs


def predict_previous_relations_head_and_tail(outputs, classifier, head_encodings, tail_encodings, predicted_father_ids=None, golden_previous_ids=None, golden_previous_labels=None, loss_ratio=0.3):
    """
    :param classifier: a specific nn.Module, PairClassifier by default
    :param encodings: (batch_size, max_num_node, hidden_dim)
    :param father_node_logit_scores: (batch_size, max_num_node, max_num_node)
    :param golden_previous_ids: (batch_size, max_num_node,)
    :param golden_previous_labels: (batch_size, max_num_node, int:2or3)
    """

    if golden_previous_ids is None:  # inference
        # fathers = torch.argmax(father_node_logit_scores, dim=2)
        fathers = predicted_father_ids.tolist() if type(predicted_father_ids) == torch.Tensor else predicted_father_ids
        ###### previous_id_list = father_id_to_previous_id(fathers)
        previous_id_list =  father_id_to_previous_id([[idx+1 for idx in father_ids] for father_ids in fathers])
        previous_id_list = torch.tensor(previous_id_list).to(encodings.device)

        # add previous_ids to the outputs at inference time
        if "previous_ids" not in outputs.keys() or outputs["previous_ids"] is None:
            ###### predicted_previous = father_id_to_previous_id(predicted_links[:, 1:].detach().tolist())
            # predicted_previous = predicted_previous[:, 1:]
            ###### predicted_previous = torch.tensor(predicted_previous)[:, 1:] - 1
            outputs["previous_ids"] = (previous_id_list[:, 1:] - 1).detach()
    
    else: # train, teacher forcing
        previous_id_list = golden_previous_ids

    (previous_relations_logit_scores, previous_loss) = classifier(head_encodings, tail_encodings, previous_id_list, golden_previous_labels)
    # print(previous_relations_logit_scores)
    # print(previous_loss)
    # if previous_loss is not None:
    #     return {
    #         "previous_loss": previous_loss,
    #         "previous_labels": torch.argmax(previous_relations_logit_scores, dim=-1)
    #     }
    # else:
    #     return {"previous_labels": torch.argmax(previous_relations_logit_scores, dim=-1)}
    if previous_loss is not None and "loss" in outputs.keys():
        outputs["original_loss"] = outputs["loss"].clone()
        outputs["loss"] = loss_ratio * previous_loss + (1-loss_ratio) * outputs["loss"]
        # outputs.update(previous_relation_outputs)
        outputs["previous_loss"] = previous_loss
    outputs["previous_labels"] = torch.argmax(previous_relations_logit_scores, dim=-1)
    # return {
    #         "previous_loss": previous_loss,
    #         "previous_labels": torch.argmax(previous_relations_logit_scores, dim=-1)
    #     }
    return outputs


def predict_previous_relations_forward_with_graph_encodings(outputs, classifier, encodings, predicted_father_ids=None, golden_previous_ids=None, golden_previous_labels=None, loss_ratio=0.3):
    """
    :param classifier: a specific nn.Module, PairClassifier by default
    :param encodings: (batch_size, max_num_node, hidden_dim)
    :param father_node_logit_scores: (batch_size, max_num_node, max_num_node)
    :param golden_previous_ids: (batch_size, max_num_node,)
    :param golden_previous_labels: (batch_size, max_num_node, int:2or3)
    """

    if golden_previous_ids is None:  # inference
        # fathers = torch.argmax(father_node_logit_scores, dim=2)
        fathers = predicted_father_ids.tolist() if type(predicted_father_ids) == torch.Tensor else predicted_father_ids
        ###### previous_id_list = father_id_to_previous_id(fathers)
        previous_id_list =  father_id_to_previous_id([[idx+1 for idx in father_ids] for father_ids in fathers])
        previous_id_list = torch.tensor(previous_id_list).to(encodings.device)

        # add previous_ids to the outputs at inference time
        if "previous_ids" not in outputs.keys() or outputs["previous_ids"] is None:
            ###### predicted_previous = father_id_to_previous_id(predicted_links[:, 1:].detach().tolist())
            # predicted_previous = predicted_previous[:, 1:]
            ###### predicted_previous = torch.tensor(predicted_previous)[:, 1:] - 1
            outputs["previous_ids"] = (previous_id_list[:, 1:] - 1).detach()
    
    else: # train, teacher forcing
        previous_id_list = golden_previous_ids

    (previous_relations_logit_scores, previous_loss) = classifier.forward_with_graph_encodings(encodings, previous_id_list, golden_previous_labels)
    
    # print(outputs)

    if previous_loss is not None and "loss" in outputs.keys():
        outputs["original_loss"] = outputs["loss"].clone()
        outputs["loss"] = loss_ratio * previous_loss + (1-loss_ratio) * outputs["loss"]
        # outputs.update(previous_relation_outputs)
        outputs["previous_loss"] = previous_loss
    outputs["previous_labels"] = torch.argmax(previous_relations_logit_scores, dim=-1)
    # return {
    #         "previous_loss": previous_loss,
    #         "previous_labels": torch.argmax(previous_relations_logit_scores, dim=-1)
    #     }
    return outputs


def common_collate_fn(batch):

    ids = [instance["id"] for instance in batch]
    node_modal = [instance["Node_modal"] for instance in batch]

    meta = {
        "ids": ids,
        "node_modal": node_modal,
        "golden_parent_ids": [instance["Father"] for instance in batch],
        "golden_parent_relations": [instance["Father_Relation"] for instance in batch],
        "golden_previous_ids": [instance["Previous"] for instance in batch],
        "golden_previous_relations": [instance["Previous_Relation"] for instance in batch],
    }

    if "tokenized_document" in batch[0].keys():  # args.text_encoder_type == "concate"
        max_len = max([len(instance["tokenized_document"]) for instance in batch])
        input_ids = [instance["tokenized_document"] + [0] * (max_len - len(instance["tokenized_document"])) for instance in batch]
        input_mask = [[1.0] * len(instance["tokenized_document"]) + [0.0] * (max_len - len(instance["tokenized_document"])) 
                      for instance in batch]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.float)
        
    else:
        max_len = max([len(paragraph) for instance in batch for paragraph in instance["input_ids"]])
        input_ids = [[paragraph + [0] * (max_len - len(paragraph)) for paragraph in instance["input_ids"]] for instance in batch]
        input_mask = [[[1.0] * len(paragraph) + [0.0] * (max_len - len(paragraph)) for paragraph in instance["input_ids"]]
                      for instance in batch]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.float)
    
        
    """max_len = max([len(paragraph) for instance in batch for paragraph in instance["input_ids"]])
    input_ids = [[paragraph + [0] * (max_len - len(paragraph)) for paragraph in instance["input_ids"]] for instance in
                 batch]
    input_mask = [[[1.0] * len(paragraph) + [0.0] * (max_len - len(paragraph)) for paragraph in instance["input_ids"]]
                  for instance in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)"""

    previous_ids = [ [0] + [idx+1 for idx in instance["Previous"]] for instance in batch]
    previous_labels = [instance["Previous_Relation_ids"] for instance in batch]
    """previous_labels = [[idx if (idx != 3 or previous_ids[i][j + 1] == 0) else 1 for j, idx in enumerate(lst)] for i, lst in enumerate(previous_labels)]"""
    previous_ids = torch.tensor(previous_ids, dtype=torch.long)
    previous_labels = torch.tensor(previous_labels, dtype=torch.long)
    
    output = {"meta": meta,
              "input_ids": input_ids,
              "input_mask": input_mask,
              "golden_previous_ids": previous_ids,
              "golden_previous_labels": previous_labels,
              "htlmtag_ids": None,
              "xpath_tag_ids": None,
              "xpath_index": None,
              "xpath_length": None, 
              }

    
    edu_num = [len(instance["input_ids"]) for instance in batch]
    max_edu_num = max(edu_num)
    output.update({"edu_num": edu_num,})

    if "tokenized_document" in batch[0].keys():
        sep_index_list = [instance["sep_index_list"] for instance in batch]
        output.update({"sep_index_list": sep_index_list,})

    
    # TODO
    if "htmltag_ids" in batch[0].keys():
        htlmtag_ids = padding_matrix([instance["htmltag_ids"] for instance in batch])
        # "xpath_ids": [ [(text_to_id(t[0], self.html_tag_vocab.id2token), int(t[1])) for t in xpath] for xpath in instance["xpath"]]
        # xpath_length = [[len(xpath) for xpath in instance["xpath_ids"] ] for instance in batch]
        xpath_length = [[len(xpath) for xpath in instance["xpath_tag_ids"] ] for instance in batch]
        # xpath_tag_ids = [[ [x[0] for x in xpath] for xpath in instance["xpath_ids"] ] for instance in batch]  # TODO
        # xpath_index = [[ [x[1] for x in xpath] for xpath in instance["xpath_ids"] ] for instance in batch]
        xpath_tag_ids = [instance["xpath_tag_ids"] for instance in batch]
        # print(xpath_tag_ids)
        xpath_index = [instance["xpath_index"] for instance in batch]

        xpath_length = padding_matrix(xpath_length)
        xpath_tag_ids = padding_matrix(xpath_tag_ids)
        xpath_index = padding_matrix(xpath_index)

        htlmtag_ids = torch.tensor(htlmtag_ids, dtype=torch.long)
        # print(xpath_tag_ids)
        xpath_tag_ids = torch.tensor(xpath_tag_ids, dtype=torch.long)
        xpath_index = torch.tensor(xpath_index, dtype=torch.long)
        # print(xpath_tag_ids)
        
        output.update(
            {
                "htmltag_ids": htlmtag_ids,
                "xpath_tag_ids": xpath_tag_ids,
                "xpath_index": xpath_index,
                "xpath_length": xpath_length,
            }
        )
    return output


def padding_matrix(matrix, pad_value=0):
    """
    matrix: List[List] with (batch_size, edu_num, ) for node-level features, e.g. HTML tag
        or List[List[List]] (batch_size, edu_num, *) for node-level features with variable length, e.g. XPath
    """
    batch_size = len(matrix)
    max_edu_num = max([len(doc) for doc in matrix])
    if type(matrix[0][0]) is not list:
        return [doc + [pad_value]*(max_edu_num-len(doc)) for doc in matrix]
    else:
        max_feature_len = max([len(feature)  for doc in matrix for feature in doc])
        return [[feature + [pad_value]*(max_feature_len-len(feature)) for feature in doc] for doc in matrix]
    

def self_adaption_length(length_list, total_max_length):
    assert total_max_length > 0
    if sum(length_list) <= total_max_length:
        return length_list
    while(sum(length_list) > total_max_length):
        length_list[length_list.index(max(length_list))] -= 1
    return length_list




if __name__ == "__main__":
    print(father_id_to_previous_id([[0, 1, 2,2,2,2,1,7,7,7,7,7,1,13,13,13,13,13,13,13,0,0,0,0,0,0,]+[26]*15+[0,0]]))
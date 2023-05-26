from typing import List, Dict
from collections import defaultdict, Counter

import os

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

from metrics import ListAccuracyMetric, SubdiscoursePRFMetric, MultipleClassificationAndListAccuracyMetric, MaskedListAccuracyMetric, MultipleConditionListAccuracyMetric
from processor import FATHER_RELATION_dict, PREVIOUS_RELATION_dict, Vocab
from utils import flatten_list

# FATHER_RELATION_dict = {1: "Elaboration", 2: "Topic&Key", 3: "Attribute", 4: "Explanation", 5: "Literature", 6:"Caption", 7:"Placeholder", 0: "NA"}
FATHER_RELATION_dict = {1: "Elaboration", 2: "Topic&Tag", 3: "Attribute", 4: "Explanation", 5: "Literature", 6:"Caption", 7:"Placeholder", 0: "NA"}
PREVIOUS_RELATION_dict = {1: "Narration", 2: "List", 3:"Parallel&Contrast", 4:"Topic_Transition", 5:"Break", 0: "NA"}
FATHER_RELATION_FINE_dict = {1:'Summary', 2:'Aggregation', 3:'Paraphrase', 4:'Partial',
5:'Tag', 6:'Topic', 7:'Content', 8:'Title', 9:'Quotation', 10:'Key-Value', 11:'Claim-Supporting',
12:'Question-Answer', 13:'Descriptive_Caption', 14:'Meta_Caption', 15:'Placeholder', 16:'Literature', 0:'NA'}



def draw_classification_confusion_matrix(document_records: List[Dict],
    output_dir="confusion.png",
    print_number=True,
    golden_link_only=True,
    diagnal=False):

    # plt.figure(figsize=(80, 100))
    
    parent_labels_name = list(FATHER_RELATION_dict.values())[:-1]
    previous_labels_name = list(PREVIOUS_RELATION_dict.values())[:-1]

    total_data_list = {
    "golden parent ids": [],
    "predict parent ids": [],
    "golden parent relations": [],
    "predict parent relations": [],
    "golden previous ids": [],
    "predict previous ids": [],
    "golden previous relations": [],
    "predict previous relations": []
    }

    for document_record in document_records:
        for k in total_data_list.keys():
            total_data_list[k] += document_record[k]

    # all_dict = [list(d.values()) for d in lst]
    # all_dict = [y for x in all_dict for y in x]
    # confusion_tuples = [(s.split(" -> ")[0], s.split(" -> ")[1]) for s in all_dict]
    parent_confusion_tuples = []
    previous_confusion_tuples = []
    for i in range(len(total_data_list["golden parent ids"])):
        if (not golden_link_only) or (total_data_list["golden parent ids"][i] == total_data_list["predict parent ids"][i]):
            if diagnal or (total_data_list["golden parent relations"][i] != total_data_list["predict parent relations"][i]):
                parent_confusion_tuples.append((total_data_list["golden parent relations"][i], total_data_list["predict parent relations"][i]))
        # elif not golden_link_only:
        #     confusion_tuples.append((total_data_list["golden parent relations"][i], total_data_list["predict parent relations"][i]))

        if (not golden_link_only) or (total_data_list["golden previous ids"][i] == total_data_list["predict previous ids"][i]):
            if diagnal or (total_data_list["golden previous relations"][i] != total_data_list["predict previous relations"][i]):
                previous_confusion_tuples.append((total_data_list["golden previous relations"][i], total_data_list["predict previous relations"][i]))
        # elif not golden_link_only:
        #     confusion_tuples.append((total_data_list["golden previous relations"][i], total_data_list["predict previous relations"][i]))

    parent_y_true = [t[0] for t in parent_confusion_tuples]
    parent_y_pred = [t[1] for t in parent_confusion_tuples]
    previous_y_true = [t[0] for t in previous_confusion_tuples]
    previous_y_pred = [t[1] for t in previous_confusion_tuples]

    # y_true = [int(j) for i, j in enumerate(y_true) if (int(j) < 80 and int(y_pred[i]) < 80)]  # for link
    # y_pred = [int(j) for i, j in enumerate(y_pred) if (int(j) < 80 and int(y_true[i]) < 80)]  # for link

    # (1) parent confusion matrix
    # C = confusion_matrix(y_true, y_pred, labels=list(range(80)))
    # C = confusion_matrix(y_true, y_pred, labels=list(set(y_true+y_pred)))
    C = confusion_matrix(parent_y_true, parent_y_pred, labels=parent_labels_name)
    print(C)
    # print(list(set(y_true+y_pred)))
    print(parent_labels_name)

    # plt.figure(figsize=(2000, 2000))
    plt.figure()
    plt.matshow(C, cmap=plt.cm.Reds)
    # plt.colorbar()
    
    if print_number:
        for i in range(len(C)):
            for j in range(len(C)):
                plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    
    num_local = np.array(range(len(parent_labels_name)))
    plt.xticks(num_local, parent_labels_name, fontsize=30, rotation=45)
    plt.yticks(num_local, parent_labels_name, fontsize=30, rotation=0)

    plt.tick_params(labelsize=5)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.tight_layout()
    # plt.subplots(constrained_layout=True)
    plt.savefig("parent_" + output_dir)

    # (1) previous confusion matrix
    C = confusion_matrix(previous_y_true, previous_y_pred, labels=previous_labels_name)
    print(C)
    # print(list(set(y_true+y_pred)))
    print(previous_labels_name)

    # plt.figure(figsize=(2000, 2000))
    plt.figure()
    plt.matshow(C, cmap=plt.cm.Reds)
    # plt.colorbar()
    
    if print_number:
        for i in range(len(C)):
            for j in range(len(C)):
                plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    
    num_local = np.array(range(len(previous_labels_name)))
    plt.xticks(num_local, previous_labels_name, fontsize=30, rotation=45)
    plt.yticks(num_local, previous_labels_name, fontsize=30, rotation=0)

    plt.tick_params(labelsize=5)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.tight_layout()
    # plt.subplots(constrained_layout=True)
    plt.savefig("previous_" + output_dir)


# {
#     # "parent_acc": ListAccuracyMetric(metric_name="parent_acc", dataset_name=tag),  # parent node UAS
#     "parent_uas": MaskedListAccuracyMetric(metric_name="parent_uas", dataset_name=tag),  # parent node UAS
#     # "previous_acc": ListAccuracyMetric(metric_name="previous_acc", dataset_name=tag),  # previous node UAS
#     "parent_las": MultipleConditionListAccuracyMetric(metric_name="parent_las", dataset_name=tag),
#     "previous_uas": MaskedListAccuracyMetric(metric_name="previous_uas", dataset_name=tag),  # previous node UAS, added
#     "previous_las": MultipleConditionListAccuracyMetric(metric_name="previous_las", dataset_name=tag),  # previous node acc, both previous id and previous relation should be correct
#     "previous_exact_acc": MaskedListAccuracyMetric(metric_name="previous_exact_acc", dataset_name=tag),  # if previous_id == i and previous_relation == "Continue" --> i; if previous_relation == "Break" --> -1
#     # "parent_acc_wo_figure": ListAccuracyMetric(metric_name="parent_acc_wo_figure", dataset_name=tag),
#     "parent_acc_wo_figure": MaskedListAccuracyMetric(metric_name="parent_acc_wo_figure", dataset_name=tag),
#     "parent_acc_attributed": MultipleClassificationAndListAccuracyMetric(metric_name="parent_acc_attributed", dataset_name=tag),  # parent node LAS
#     "discourse_acc": SubdiscoursePRFMetric(metric_name="discourse_acc", dataset_name=tag), 
# }

# if model_implemented_funtions["parent_ids"]:
#     # UAS, unlabeled attachment score
#     evaluation_metrics["parent_uas"].update(
#         golden=total_golden_parent_ids,
#         pred=total_pred_parent_ids,
#     )
#     # UAS, unlabeled attachment score, without considering figures
#     evaluation_metrics["parent_acc_wo_figure"].update(
#         golden=total_golden_parent_ids,
#         pred=total_pred_parent_ids,
#         mask=[1 if (total_node_modals[i] != "Figure" and total_node_modals[i] != "Figure&Title") else 0
#               for i, id in enumerate(total_node_modals)],
#     )
#     if model_implemented_funtions["parent_relations"]:
#         evaluation_metrics["parent_las"].update(
#             goldens=(total_golden_parent_ids, total_golden_parent_relations),
#             preds=(total_pred_parent_ids, total_pred_parent_relations),
#         )
#         # LAS, labeled attachment score
#         evaluation_metrics["parent_acc_attributed"].update(
#             parent_goldens=total_golden_parent_ids,
#             parent_preds=total_pred_parent_ids,
#             parent_relation_goldens=total_golden_parent_relations,
#             parent_relation_preds=total_pred_parent_relations,
#             id2classmapper=FATHER_RELATION_dict,
#         )
# if model_implemented_funtions["previous_ids"] and model_implemented_funtions["previous_relations"]:
#     # segmentation acc
#     """evaluation_metrics["previous_acc"].update(
#         goldens=total_golden_previous_relations,
#         preds=total_pred_previous_relations,
#         mask_id=3)"""
#     evaluation_metrics["previous_uas"].update(
#         golden=total_golden_previous_ids,
#         pred=total_pred_previous_ids,
#     )
#     evaluation_metrics["previous_las"].update(
#         goldens=(total_golden_previous_ids, total_golden_previous_relations),
#         preds=(total_pred_previous_ids, total_pred_previous_relations),
#     )
#     evaluation_metrics["previous_exact_acc"].update(
#         golden=[node_id if label != "Break" else -1 for node_id, label in zip(total_golden_previous_ids, total_golden_previous_relations)],
#         pred=[node_id if label != "Break" else -1 for node_id, label in zip(total_pred_previous_ids, total_pred_previous_relations)],
#     )
# if model_implemented_funtions["parent_ids"] and model_implemented_funtions["previous_ids"] and model_implemented_funtions["previous_relations"]:
#     evaluation_metrics["discourse_acc"].update(
#         parent_goldens=total_golden_parent_ids,
#         parent_preds=total_pred_parent_ids,
#         previous_goldens=total_golden_previous_relations,
#         previous_preds=total_pred_previous_relations,
#         length_of_discourse=length_of_documents)
# metrics = {metric_name: evaluation_metrics[metric_name].value for metric_name in evaluation_metrics.keys()}


# (4) calculate metrics
def calculate_metrics(total_golden_parent_ids, total_pred_parent_ids, total_golden_parent_relations, total_pred_parent_relations,
    total_golden_previous_ids, total_pred_previous_ids, total_golden_previous_relations, total_pred_previous_relations,) -> Dict:
    tag = "analysis"
    evaluation_metrics = {
        # "parent_acc": ListAccuracyMetric(metric_name="parent_acc", dataset_name=tag),  # parent node UAS
        "parent_uas": MaskedListAccuracyMetric(metric_name="parent_uas", dataset_name=tag),  # parent node UAS
        # "previous_acc": ListAccuracyMetric(metric_name="previous_acc", dataset_name=tag),  # previous node UAS
        "parent_las": MultipleConditionListAccuracyMetric(metric_name="parent_las", dataset_name=tag),
        "previous_uas": MaskedListAccuracyMetric(metric_name="previous_uas", dataset_name=tag),  # previous node UAS, added
        "previous_las": MultipleConditionListAccuracyMetric(metric_name="previous_las", dataset_name=tag),  # previous node acc, both previous id and previous relation should be correct
        "previous_exact_acc": MaskedListAccuracyMetric(metric_name="previous_exact_acc", dataset_name=tag),  # if previous_id == i and previous_relation == "Continue" --> i; if previous_relation == "Break" --> -1
        # "parent_acc_wo_figure": ListAccuracyMetric(metric_name="parent_acc_wo_figure", dataset_name=tag),
        ########"parent_acc_wo_figure": MaskedListAccuracyMetric(metric_name="parent_acc_wo_figure", dataset_name=tag),
        ####"parent_acc_attributed": MultipleClassificationAndListAccuracyMetric(metric_name="parent_acc_attributed", dataset_name=tag),  # parent node LAS
        ########"discourse_acc": SubdiscoursePRFMetric(metric_name="discourse_acc", dataset_name=tag), 
    }

    if total_pred_parent_ids:
        # UAS, unlabeled attachment score
        evaluation_metrics["parent_uas"].update(
            golden=total_golden_parent_ids,
            pred=total_pred_parent_ids,
        )
        # UAS, unlabeled attachment score, without considering figures
        # evaluation_metrics["parent_acc_wo_figure"].update(
        #     golden=total_golden_parent_ids,
        #     pred=total_pred_parent_ids,
        #     mask=[1 if (total_node_modals[i] != "Figure" and total_node_modals[i] != "Figure&Title") else 0
        #           for i, id in enumerate(total_node_modals)],
        # )
        if total_pred_parent_relations:
            evaluation_metrics["parent_las"].update(
                goldens=(total_golden_parent_ids, total_golden_parent_relations),
                preds=(total_pred_parent_ids, total_pred_parent_relations),
            )
            # LAS, labeled attachment score
            """evaluation_metrics["parent_acc_attributed"].update(
                parent_goldens=total_golden_parent_ids,
                parent_preds=total_pred_parent_ids,
                parent_relation_goldens=total_golden_parent_relations,
                parent_relation_preds=total_pred_parent_relations,
                id2classmapper=FATHER_RELATION_dict,
            )"""
    if total_pred_previous_ids:
        # segmentation acc
        """evaluation_metrics["previous_acc"].update(
            goldens=total_golden_previous_relations,
            preds=total_pred_previous_relations,
            mask_id=3)"""
        evaluation_metrics["previous_uas"].update(
            golden=total_golden_previous_ids,
            pred=total_pred_previous_ids,
        )
        if total_pred_previous_relations:
            evaluation_metrics["previous_las"].update(
                goldens=(total_golden_previous_ids, total_golden_previous_relations),
                preds=(total_pred_previous_ids, total_pred_previous_relations),
            )
            evaluation_metrics["previous_exact_acc"].update(
                golden=[node_id if label != "Break" else -1 for node_id, label in zip(total_golden_previous_ids, total_golden_previous_relations)],
                pred=[node_id if label != "Break" else -1 for node_id, label in zip(total_pred_previous_ids, total_pred_previous_relations)],
            )
    # if model_implemented_funtions["parent_ids"] and model_implemented_funtions["previous_ids"] and model_implemented_funtions["previous_relations"]:
    #     evaluation_metrics["discourse_acc"].update(
    #         parent_goldens=total_golden_parent_ids,
    #         parent_preds=total_pred_parent_ids,
    #         previous_goldens=total_golden_previous_relations,
    #         previous_preds=total_pred_previous_relations,
    #         length_of_discourse=length_of_documents)
    metrics = {metric_name: evaluation_metrics[metric_name].value for metric_name in evaluation_metrics.keys()}
    return metrics




# Main function
def analysis_all(record_file, save_file, epoch, tag="test", step=1, propotional_step=0.05, confusion_matrix_output="confusion.png"):
    if type(tag) == list:    # Update, to support analysis multiple dataset split together, e.g., ["dev", "test"]
        record_file_lines = []
        document_records = []
        for t in tag:
            print(f"load and parse {t} set at Epoch {epoch}")
            record_file_line_single_split = load_record_file(record_file, epoch, t)
            # print(record_file_lines)
            document_record_single_split = parse_record_file(record_file_line_single_split, t)
            record_file_lines += record_file_line_single_split
            document_records += document_record_single_split

    else:
        print(f"load and parse {tag} set at Epoch {epoch}")
        record_file_lines = load_record_file(record_file, epoch, tag)
        document_records = parse_record_file(record_file_lines, tag)


    group_by_num_edu                 , counter = group_document_by_num_edu(document_records, step=step)
    group_by_num_break               , counter = group_document_by_num_break(document_records, step=5)
    group_by_edu_id                  , counter = group_edu_by_edu_id(document_records, step=step)
    group_by_dependency_link_distance, counter = group_edu_by_dependency_link_distance(document_records, step=5)
    group_by_golden_parent_relation  , counter = group_edu_by_golden_parent_relation(document_records)
    group_by_golden_previous_relation, counter = group_edu_by_golden_previous_relation(document_records)

    group_by_num_break_propotional   , counter = group_document_by_num_break_propotional(document_records, step=0.1)  #step=propotional_step)  # Update
    group_by_edu_id_propotional      , counter = group_edu_by_edu_id_propotional(document_records, step=propotional_step)  # Update


    metrics_by_num_edu                  = analisis_group_dict(group_by_num_edu)
    metrics_by_num_break                = analisis_group_dict(group_by_num_break)
    metrics_by_edu_id                   = analisis_group_dict(group_by_edu_id)
    metrics_by_dependency_link_distance = analisis_group_dict(group_by_dependency_link_distance)
    metrics_by_golden_parent_relation   = analisis_group_dict(group_by_golden_parent_relation)
    metrics_by_golden_previous_relation = analisis_group_dict(group_by_golden_previous_relation)

    metrics_by_num_break_propotional = analisis_group_dict(group_by_num_break_propotional)  # Update
    metrics_by_edu_id_propotional = analisis_group_dict(group_by_edu_id_propotional)  # Update

    print(metrics_by_num_edu)
    print(metrics_by_num_break               )
    print(metrics_by_edu_id                  )
    print(metrics_by_dependency_link_distance)
    print(metrics_by_golden_parent_relation  )
    print(metrics_by_golden_previous_relation)
    print(metrics_by_num_break_propotional)
    print(metrics_by_edu_id_propotional)

    structural_error_type_counter, id_error_type_counter, absolute_id_error_type_counter, relative_id_error_type_counter, structural_error_type_cases \
     = structural_error_distribution(document_records)
    
    metric_list = [metrics_by_num_edu, metrics_by_num_break, metrics_by_edu_id, metrics_by_dependency_link_distance,
        metrics_by_golden_parent_relation, metrics_by_golden_previous_relation,
        structural_error_type_counter, id_error_type_counter, absolute_id_error_type_counter, relative_id_error_type_counter,
        metrics_by_num_break_propotional, metrics_by_edu_id_propotional
        ]
    metric_name_list = ["num_edu", "num_break", "edu_id", "dependency_link_distance", "golden_parent_relation", "golden_previous_relation",
        "structural_error_type_counter", "id_error_type_counter", "absolute_id_error_type_counter", "relative_id_error_type_counter", "num_break_propotional", "edu_id_propotional"]
    
    draw_classification_confusion_matrix(document_records,
        output_dir=confusion_matrix_output,
        print_number=True,
        golden_link_only=True)
    
    with open(save_file, "w", encoding="utf-8") as save_f:
        save_dict = {name: metric for metric, name in zip(metric_list, metric_name_list)}
        # save_f.write("\n\n".join([name + "\n" + str(metric) for metric, name in zip(metric_list, metric_name_list)]))
        save_f.write(str(save_dict))
        save_f.write("\n\n")
        save_f.write("\n\n".join([name + "\n" + str(metric) for metric, name in zip(metric_list, metric_name_list)]))
        save_f.write("structural_error_type_cases" + "\n" + str(structural_error_type_cases))

    return metrics_by_num_edu, metrics_by_num_break, metrics_by_edu_id, metrics_by_dependency_link_distance, metrics_by_golden_parent_relation, metrics_by_golden_previous_relation, \
        structural_error_type_counter, id_error_type_counter, metrics_by_num_break_propotional, metrics_by_edu_id_propotional



# (1) read record file and locate specific epoch and datasplit
def load_record_file(record_file, epoch, tag="test", cache_file=""):
    r_file_lines = []
    with open(record_file, "r", encoding="utf-8") as r_file:
        epoch_flag = False
        datasplit_flag = False
        
        for line in r_file:

            if epoch_flag and datasplit_flag:
                if line.startswith("=" * 20):
                    break
                r_file_lines.append(line)

            if epoch_flag and not datasplit_flag:
                if line.startswith(f"Evaluate on {tag} set" + "\t"):
                    datasplit_flag = True
                    r_file_lines.append(line)
                else:
                    r_file_lines = []

            if not epoch_flag:
                if line.startswith("=" * 20 + f"Epoch = {epoch}" + "=" * 20 + "\n"):
                    epoch_flag = True
                    r_file_lines.append(line)
            # else:
            #     epoch_flag = False
        # r_file_lines = r_file.read().strip().split("\n")
    return r_file_lines
            

# (2) parse List of str (lines in record fine) into data structure List of document record (Dict)
def parse_record_file(record_file_lines: List[str], tag: str="test") -> List[Dict[str, str or List]]:
    # document_id_list = []
    start_strings = {
        f"Evaluate on {tag} set" + "\t" + "id =": [],  # document_id_list
        "golden parent ids": [],
        "predict parent ids": [],
        "golden parent relations": [],
        "predict parent relations": [],
        "golden previous ids": [],
        "predict previous ids": [],
        "golden previous relations": [],
        "predict previous relations": [],
    }

    for line in record_file_lines:
        for k, v in start_strings.items():
            if line.strip().startswith(k):
                value = line[len(k):][1:]
                if k != f"Evaluate on {tag} set" + "\t" + "id =":
                    value = eval(value)
                v.append(value)
    
    assert len(start_strings[f"Evaluate on {tag} set" + "\t" + "id ="]) == len(start_strings["golden parent ids"])
    assert len(start_strings[f"Evaluate on {tag} set" + "\t" + "id ="]) == len(start_strings["predict parent ids"])
    assert len(start_strings[f"Evaluate on {tag} set" + "\t" + "id ="]) == len(start_strings["golden parent relations"])
    assert len(start_strings[f"Evaluate on {tag} set" + "\t" + "id ="]) == len(start_strings["predict parent relations"])
    assert len(start_strings[f"Evaluate on {tag} set" + "\t" + "id ="]) == len(start_strings["golden previous ids"])
    assert len(start_strings[f"Evaluate on {tag} set" + "\t" + "id ="]) == len(start_strings["predict previous ids"])
    assert len(start_strings[f"Evaluate on {tag} set" + "\t" + "id ="]) == len(start_strings["golden previous relations"])
    assert len(start_strings[f"Evaluate on {tag} set" + "\t" + "id ="]) == len(start_strings["predict previous relations"])
    print(f"num documents = {len(start_strings['golden parent ids'])}")

    document_records = []

    for i, document_id in enumerate(start_strings[f"Evaluate on {tag} set" + "\t" + "id ="]):
        document_records.append(
            {
                "id": document_id,
                "golden parent ids": start_strings["golden parent ids"][i],
                "predict parent ids": start_strings["predict parent ids"][i],
                "golden parent relations": start_strings["golden parent relations"][i],
                "predict parent relations": start_strings["predict parent relations"][i],
                "golden previous ids": start_strings["golden previous ids"][i],
                "predict previous ids": start_strings["predict previous ids"][i],
                "golden previous relations": start_strings["golden previous relations"][i],
                "predict previous relations": start_strings["predict previous relations"][i],

                "num_edu": len(start_strings["golden parent ids"][i]),
                "num_break": start_strings["golden previous relations"][i].count('Break'),
            }
        )
    
    for document_record in document_records:  # Padding
        document_record["golden parent relations"] = [i  if i != "Topic&Key" else "Topic&Tag"  for i in document_record["golden parent relations"]]
        document_record["predict parent relations"] = [i  if i != "Topic&Key" else "Topic&Tag"  for i in document_record["predict parent relations"]]

    return document_records



# (3) document-level and EDU-level data grouping
def counter_to_propotion(counter):
    counter = dict(counter)
    v_sum = sum(list(counter.values()))
    counter = {k:v/v_sum for k, v in counter.items()}
    counter = dict(sorted(counter.items(), key=lambda x:x[0]))
    print(counter)
    return counter

KEY_NAMES = [
    "golden parent ids",
    "predict parent ids",
    "golden parent relations",
    "predict parent relations",
    "golden previous ids",
    "predict previous ids",
    "golden previous relations",
    "predict previous relations"
]

# document-level grouping
def group_document_by_num_edu(document_records: List[Dict], step=1, max_interval=100) -> Dict[int, Dict[str, List]]:
    grouped_dict = {}
    counter = defaultdict(int)
    for document_record in document_records:
        num_edu = document_record["num_edu"]
        num_edu = (num_edu // step) * step
        num_edu = min(num_edu, max_interval)
        '''if num_edu == max_interval:
            num_edu = f">{max_interval}"'''
        counter[num_edu] += 1
        if num_edu not in grouped_dict.keys():
            grouped_dict[num_edu] = {}
            for k in KEY_NAMES:
                grouped_dict[num_edu][k]: List[List] = [document_record[k]]
            grouped_dict[num_edu]["id"]: List[str] = [document_record["id"]]
        else:
            for k in KEY_NAMES:
                grouped_dict[num_edu][k].append(document_record[k])
            grouped_dict[num_edu]["id"].append(document_record["id"])
    return grouped_dict, counter_to_propotion(counter)


def group_document_by_num_break(document_records: List[Dict], step=1, max_interval=40) -> Dict[int, Dict[str, List]]:
    grouped_dict = {}
    counter = defaultdict(int) 
    for document_record in document_records:
        num_edu = document_record["num_break"]
        num_edu = (num_edu // step) * step 
        num_edu = min(num_edu, max_interval) 
        '''if num_edu == max_interval:
            num_edu = f">{max_interval}"'''
        counter[num_edu] += 1
        if num_edu not in grouped_dict.keys():
            grouped_dict[num_edu] = {}
            for k in KEY_NAMES:
                grouped_dict[num_edu][k]: List[List] = [document_record[k]]
            grouped_dict[num_edu]["id"]: List[str] = [document_record["id"]]
        else:
            for k in KEY_NAMES:
                grouped_dict[num_edu][k].append(document_record[k])
            grouped_dict[num_edu]["id"].append(document_record["id"])
    return grouped_dict, counter_to_propotion(counter)

def group_document_by_num_break_propotional(document_records: List[Dict], step=0.05) -> Dict[float, Dict[str, List]]:
    grouped_dict = {}
    counter = defaultdict(int)
    for document_record in document_records:
        num_edu = document_record["num_break"]
        num_edu_propotional = num_edu / document_record["num_edu"]
        num_edu_propotional = round((num_edu_propotional // step) * step, 2)
        counter[num_edu_propotional] += 1

        if num_edu_propotional not in grouped_dict.keys():
            grouped_dict[num_edu_propotional] = {}
            for k in KEY_NAMES:
                grouped_dict[num_edu_propotional][k]: List[List] = [document_record[k]]
            grouped_dict[num_edu_propotional]["id"]: List[str] = [document_record["id"]]
        else:
            for k in KEY_NAMES:
                grouped_dict[num_edu_propotional][k].append(document_record[k])
            grouped_dict[num_edu_propotional]["id"].append(document_record["id"])
    return grouped_dict, counter_to_propotion(counter)


# EDU-level grouping
def group_edu_by_edu_id(document_records: List[Dict], step=1, max_interval=140) -> Dict[int, Dict[str, List]]:
    grouped_dict = {}
    counter = defaultdict(int)
    for document_record in document_records:
        num_edu = len(document_record["golden parent ids"])
        
        for i in range(num_edu):
            edu_id = (i // step) * step
            edu_id = min(edu_id, max_interval)
            '''if num_edu == max_interval:
                num_edu = f">{max_interval}"'''
            counter[edu_id] += 1

            if edu_id not in grouped_dict.keys():
                grouped_dict[edu_id] = {}
                for k in KEY_NAMES:
                    grouped_dict[edu_id][k]: List[int] = [document_record[k][i]]
                grouped_dict[edu_id]["id"]: List[str] = [document_record["id"]]
            else:
                for k in KEY_NAMES:
                    grouped_dict[edu_id][k].append(document_record[k][i])
                grouped_dict[edu_id]["id"].append(document_record["id"])
    return grouped_dict, counter_to_propotion(counter)


def group_edu_by_edu_id_propotional(document_records: List[Dict], step=0.05) -> Dict[float, Dict[str, List]]:
    grouped_dict = {}
    counter = defaultdict(int)
    for document_record in document_records:
        num_edu = len(document_record["golden parent ids"])
        
        for i in range(num_edu):
            edu_id_propotional = i / document_record["num_edu"]
            edu_id_propotional = round((edu_id_propotional // step) * step, 2)
            counter[edu_id_propotional] += 1

            if edu_id_propotional not in grouped_dict.keys():
                grouped_dict[edu_id_propotional] = {}
                for k in KEY_NAMES:
                    grouped_dict[edu_id_propotional][k]: List[int] = [document_record[k][i]]
                grouped_dict[edu_id_propotional]["id"]: List[str] = [document_record["id"]]
            else:
                for k in KEY_NAMES:
                    grouped_dict[edu_id_propotional][k].append(document_record[k][i])
                grouped_dict[edu_id_propotional]["id"].append(document_record["id"])
    return grouped_dict, counter_to_propotion(counter)


def group_edu_by_dependency_link_distance(document_records: List[Dict], step=1, max_interval=80) -> Dict[int, Dict[str, List]]:
    grouped_dict = {}
    counter = defaultdict(int)
    for document_record in document_records:
        num_edu = len(document_record["golden parent ids"])
        
        for i in range(num_edu):
            golden_parent_id = document_record["golden parent ids"][i]
            parent_link_distance = i - golden_parent_id if int(golden_parent_id) != -1 else -1
            parent_link_distance = (parent_link_distance // step) * step if parent_link_distance != -1 else -1

            parent_link_distance = min(parent_link_distance, max_interval)
            '''if num_edu == max_interval:
                num_edu = f">{max_interval}"'''
            counter[parent_link_distance] += 1

            if parent_link_distance not in grouped_dict.keys():
                grouped_dict[parent_link_distance] = {}
                for k in KEY_NAMES:
                    grouped_dict[parent_link_distance][k]: List[int] = [document_record[k][i]]
                grouped_dict[parent_link_distance]["id"]: List[str] = [document_record["id"]]
            else:
                for k in KEY_NAMES:
                    grouped_dict[parent_link_distance][k].append(document_record[k][i])
                grouped_dict[parent_link_distance]["id"].append(document_record["id"])
    return grouped_dict, counter_to_propotion(counter)


def group_edu_by_golden_parent_relation(document_records: List[Dict]) -> Dict[str, Dict[str, List]]:
    grouped_dict = {}
    counter = defaultdict(int)
    for document_record in document_records:
        num_edu = len(document_record["golden parent ids"])
        
        for i in range(num_edu):
            golden_parent_relation = document_record["golden parent relations"][i]
            counter[golden_parent_relation] += 1

            if golden_parent_relation not in grouped_dict.keys():
                grouped_dict[golden_parent_relation] = {}
                for k in KEY_NAMES:
                    grouped_dict[golden_parent_relation][k]: List[int] = [document_record[k][i]]
                grouped_dict[golden_parent_relation]["id"]: List[str] = [document_record["id"]]
            else:
                for k in KEY_NAMES:
                    grouped_dict[golden_parent_relation][k].append(document_record[k][i])
                grouped_dict[golden_parent_relation]["id"].append(document_record["id"])
    return grouped_dict, counter_to_propotion(counter)

def group_edu_by_golden_previous_relation(document_records: List[Dict]) -> Dict[str, Dict[str, List]]:
    grouped_dict = {}
    counter = defaultdict(int)
    for document_record in document_records:
        num_edu = len(document_record["golden parent ids"])
        
        for i in range(num_edu):
            golden_previous_relation = document_record["golden previous relations"][i]
            counter[golden_previous_relation] += 1

            if golden_previous_relation not in grouped_dict.keys():
                grouped_dict[golden_previous_relation] = {}
                for k in KEY_NAMES:
                    grouped_dict[golden_previous_relation][k]: List[int] = [document_record[k][i]]
                grouped_dict[golden_previous_relation]["id"]: List[str] = [document_record["id"]]
            else:
                for k in KEY_NAMES:
                    grouped_dict[golden_previous_relation][k].append(document_record[k][i])
                grouped_dict[golden_previous_relation]["id"].append(document_record["id"])
    return grouped_dict, counter_to_propotion(counter)


# KEY_NAMES = [
#     "golden parent ids",
#     "predict parent ids",
#     "golden parent relations",
#     "predict parent relations",
#     "golden previous ids",
#     "predict previous ids",
#     "golden previous relations",
#     "predict previous relations"
# ]

def flatten_group(group: Dict[str, List]):
    total_golden_parent_ids      = flatten_list(group["golden parent ids"])
    total_pred_parent_ids        = flatten_list(group["predict parent ids"])
    total_golden_parent_relations= flatten_list(group["golden parent relations"])
    total_pred_parent_relations  = flatten_list(group["predict parent relations"])
    total_golden_previous_ids    = flatten_list(group["golden previous ids"])
    total_pred_previous_ids      = flatten_list(group["predict previous ids"])
    total_golden_previous_relations= flatten_list(group["golden previous relations"])
    total_pred_previous_relations= flatten_list(group["predict previous relations"])
    return total_golden_parent_ids, total_pred_parent_ids, total_golden_parent_relations, total_pred_parent_relations, total_golden_previous_ids, total_pred_previous_ids, total_golden_previous_relations, total_pred_previous_relations



def analisis_group_dict(grouped_dict: Dict[int or str, Dict[str, List]]) -> Dict[str, Dict]:
    # grouped_dict = group_document_by_num_edu(document_records)  # Dict[Dict]
    metric_dict = {}  # Dict[metric_name: Dict[group_key: metric_value]]

    for group_key, group in grouped_dict.items():
        golden_and_pred_fields = flatten_group(group)
        metrics = calculate_metrics(*golden_and_pred_fields)  # Dict[metric_name: metric_value]
        # metrics = {metric_name: evaluation_metrics[metric_name].value for metric_name in evaluation_metrics.keys()}
        for metric_name, metric_value in metrics.items():
            if metric_name not in metric_dict.keys():
                metric_dict[metric_name] = {group_key : metric_value}
            else:
                metric_dict[metric_name][group_key] = metric_value

    for metric_name, metric in metric_dict.items():
        metric_dict[metric_name] = dict(sorted(metric.items(), key=lambda x: x[0]))
    
    return metric_dict



def is_brother(current_id, predicted_id, golden_parent_ids):
    # print(current_id, predicted_id)
    current_id = int(current_id)
    predicted_id = int(predicted_id)
    golden_parent_ids = [int(i) for i in golden_parent_ids]

    father_of_current = golden_parent_ids[current_id]
    father_of_predicted = golden_parent_ids[predicted_id]

    if father_of_current == predicted_id:
        return "correct"
    if predicted_id == -1:
        return "predict dummy"

    if father_of_current == father_of_predicted:
        return "brother"

    if int(father_of_current) == -1:
        return "dummy father"
    

    
    if golden_parent_ids[father_of_current] == golden_parent_ids[predicted_id]:
        return "uncle"
    if golden_parent_ids[father_of_current] == predicted_id:
        return "grandfather"
    while(golden_parent_ids[father_of_current] != predicted_id):
        father_of_current = golden_parent_ids[father_of_current]
        if father_of_current == -1:
            break
        if golden_parent_ids[father_of_current] == predicted_id:
            return "ancestor"
    return "other"

def is_previous(current_id, predicted_id, golden_parent_ids):
    current_id = int(current_id)
    predicted_id = int(predicted_id)
    golden_parent_ids = [int(i) for i in golden_parent_ids]
    if predicted_id == current_id - 1:
        return "id-1"
    if predicted_id == 0:
        return "0"
    if predicted_id == -1:
        return "-1"
    return f"{current_id - predicted_id}"

def is_continue_brother(current_id, brother_id, golden_previous_ids, golden_previous_relations):
    current_id = int(current_id)
    brother_id = int(brother_id)
    golden_previous_ids = [int(i) for i in golden_previous_ids]
    assert brother_id != current_id
    #####print(current_id, brother_id)
    if brother_id < current_id:
        while(current_id != brother_id):
            if golden_previous_relations[current_id] == "Break":
                return False
            current_id = golden_previous_ids[current_id]
            assert current_id != -1, (current_id, brother_id)
        return True
    else:
        while(current_id != brother_id):
            if golden_previous_relations[brother_id] == "Break":
                return False
            brother_id = golden_previous_ids[brother_id]
            assert brother_id != -1, (current_id, brother_id)
        return True



def structural_error_distribution(document_records: List[Dict]) -> Dict[int, Dict[str, List]]:
    print("count structural_error_distribution ...")
    grouped_dict = {}

    structural_error_type_list = [] 
    id_error_type_list = [] 
    absolute_id_error_type_list = [] 
    relative_id_error_type_list = [] 

    structural_error_type_cases = defaultdict(list)

    for document_record in document_records:
        #####print(document_record["id"])
        did = document_record["id"]
        num_edu = document_record["num_edu"]
        golden_parent_ids = document_record["golden parent ids"]
        predict_parent_ids = document_record["predict parent ids"]
        golden_previous_ids = document_record["golden previous ids"]
        golden_previous_relations = document_record["golden previous relations"]
        
        for current_id, predicted_id in enumerate(predict_parent_ids):
            ### Structural error type
            structural_error_type = is_brother(current_id, predicted_id, golden_parent_ids)
            #####print(structural_error_type)
            if structural_error_type == "brother":
                if is_continue_brother(current_id, predicted_id, golden_previous_ids, golden_previous_relations):
                    structural_error_type = "brother-continue"
                else:
                    structural_error_type = "brother-break"
            elif structural_error_type == "uncle":
                if is_continue_brother(golden_parent_ids[current_id], predicted_id, golden_previous_ids, golden_previous_relations):
                    structural_error_type = "uncle-continue"
                else:
                    structural_error_type = "uncle-break"

            if structural_error_type != "correct":
                structural_error_type_list.append(structural_error_type)
                structural_error_type_cases[structural_error_type].append({"document_id": did, "node_id": current_id, "golden_parent": golden_parent_ids[current_id], "predicted_parent":predicted_id})

            ### ID error type
            if structural_error_type != "correct":
                id_error_type = is_previous(current_id, predicted_id, golden_parent_ids)
                id_error_type_list.append(id_error_type)

                absolute_id_error_type = predicted_id
                relative_id_error_type = current_id - predicted_id
                absolute_id_error_type_list.append(absolute_id_error_type)
                relative_id_error_type_list.append(relative_id_error_type)

    structural_error_type_counter = Counter(structural_error_type_list)
    id_error_type_counter = Counter(id_error_type_list)

    structural_error_type_counter = dict(sorted(dict(structural_error_type_counter).items(), key = lambda x: x[0]))
    id_error_type_counter = dict(sorted(dict(id_error_type_counter).items(), key = lambda x: x[0]))

    structural_error_type_counter = {k: v / sum(structural_error_type_counter.values()) for k, v in structural_error_type_counter.items()}
    id_error_type_counter = {k: v / sum(id_error_type_counter.values()) for k, v in id_error_type_counter.items()}

    print(structural_error_type_counter)
    print(id_error_type_counter)

    absolute_id_error_type_counter = Counter(absolute_id_error_type_list)
    relative_id_error_type_counter = Counter(relative_id_error_type_list)
    absolute_id_error_type_counter = counter_to_propotion(absolute_id_error_type_counter)
    relative_id_error_type_counter = counter_to_propotion(relative_id_error_type_counter)

    return structural_error_type_counter, id_error_type_counter, absolute_id_error_type_counter, relative_id_error_type_counter, structural_error_type_cases


# DO NOT use
######################################################################
def analysis_document_length(document_records: List[Dict]):
    # (1) document-level metrics – document number EDU
    # document-level, grouped by num_edu
    grouped_dict = group_document_by_num_edu(document_records)  # Dict[Dict]
    metric_dict = {}  # Dict[metric_name: Dict[group_key: metric_value]]

    for group_key, group in grouped_dict.items():
        golden_and_pred_fields = flatten_group(group)
        metrics = calculate_metrics(*golden_and_pred_fields)  # Dict[metric_name: metric_value]
        # metrics = {metric_name: evaluation_metrics[metric_name].value for metric_name in evaluation_metrics.keys()}
        for metric_name, metric_value in metrics.items():
            if metric_name not in metric_dict.keys():
                metric_dict[metric_name] = {group_key : metric_value}
            else:
                metric_dict[metric_name][group_key] = metric_value

    return metric_dict



def analysis_edu_id(document_records):
    # (2) EDU-level metrics beamed by EDU id - EDU id
    # EDU-level, grouped by EDU id
    pass

def analysis_document_break_number(document_records):
    # (3) document-level metrics – number of “Break” in document
    # document-level, grouped by num_break
    pass

def analysis_dependency_link_distance(document_records):
    # (4) EDU-level UAS beamed by golden dependency link distance - golden dependency link distance
    # EDU-level, grouped by EDU golden dependency link distance
    pass

def parent_relation_breakdown(document_records):
    # (5) EDU-level parent UAS / LAS beamed by golden parent label - golden parent label
    # EDU-level, grouped by EDU golden parent relation
    pass

def previous_relation_breakdown(document_records):
    # (6) EDU-level previous UAS / LAS beamed by golden previous label - golden previous label
    # EDU-level, grouped by EDU golden previous relation
    pass


if __name__ == "__main__":

    if not os.exists("./log_analysis"):
        os.mkdir("log_analysis")

    # 0112 All models, ["dev", "test"]
    record_file = "..."
    save_file = "..."
    epoch = 24
    analysis_all(record_file=record_file, save_file=save_file, epoch=epoch, tag=["dev", "test"], step=10)
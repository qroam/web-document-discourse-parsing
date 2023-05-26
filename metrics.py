# -*- coding: utf-8 -*-
from abc import ABC
from collections import defaultdict

from visualization import subdiscourse_parser


def safe_devide(a, b):
    if b != 0:
        return a / b
    else:
        return 0.0


def prf(golden_list, predict_list):
    num_golden = len(golden_list)
    num_predict = len(predict_list)
    num_correct = 0
    for unit in golden_list:
        if unit in predict_list:
            num_correct += 1
    recall = safe_devide(num_correct, num_golden) * 100
    precision = safe_devide(num_correct, num_predict) * 100
    if num_golden == 0 and num_predict == 0:
        precision, recall = 1.0, 1.0
    f1 = safe_devide(2 * precision * recall, precision + recall)
    return precision, recall, f1, num_golden, num_predict, num_correct

def classwise_prf(golden_list, predict_list, golden_label, predict_label):
    num_golden = len(golden_list)
    num_predict = len(predict_list)
    num_correct = 0
    num_correct_wo_label = 0

    num_golden_per_class = defaultdict(int)
    num_predict_per_class = defaultdict(int)
    num_correct_per_class = defaultdict(int)

    for unit, unit_label in zip(golden_list, golden_label):
        num_golden_per_class[unit_label] += 1
    for unit, unit_label in zip(predict_list, predict_label):
        num_predict_per_class[unit_label] += 1
    for unit, unit_label, predict_unit, predict_unit_label in zip(golden_list, golden_label, predict_list, predict_label):
        """if unit in predict_list:
            if predict_label[predict_list.index(unit)] == unit_label:
                num_correct += 1
                num_correct_per_class[unit_label] += 1
            else:
                num_correct_wo_label += 1"""
        if unit == predict_unit:
            num_correct_wo_label += 1
            if unit_label == predict_unit_label:
                num_correct += 1
                num_correct_per_class[unit_label] += 1

    # (1) micro p, r, f1, strict match (both the vertices and the relation label should be correct), i.e. Labeled Attachment Score (LAS)
    recall = safe_devide(num_correct, num_golden) * 100
    precision = safe_devide(num_correct, num_predict) * 100
    if num_golden == 0 and num_predict == 0:
        precision, recall = 1.0, 1.0
    f1 = safe_devide(2 * precision * recall, precision + recall)

    # (2) micro p, r, f1, father node match (as long as father node correct), i.e. Unlabeled Attachment Score (UAS)
    recall_wo_label = safe_devide(num_correct_wo_label, num_golden) * 100
    precision_wo_label = safe_devide(num_correct_wo_label, num_predict) * 100
    if num_golden == 0 and num_predict == 0:
        precision_wo_label, recall_wo_label = 1.0, 1.0
    f1_wo_label = safe_devide(2 * precision_wo_label * recall_wo_label, precision_wo_label + recall_wo_label)

    # (3) p, r, f1 per relation label, strict match
    recall_per_class = {label:safe_devide(num_correct_per_class[label], num_golden_per_class[label]) * 100 for label in
                           num_golden_per_class.keys()}
    precision_per_class = {label: safe_devide(num_correct_per_class[label], num_predict_per_class[label]) * 100 for label in
                           num_golden_per_class.keys()}
    for label in num_golden_per_class.keys():
        if num_golden_per_class[label] == 0 and num_predict_per_class[label] == 0:
            recall_per_class[label], precision_per_class[label] = 1.0, 1.0
    f1_per_class = {label: safe_devide(2 * precision_per_class[label] * recall_per_class[label], precision_per_class[label] + recall_per_class[label])
                    for label in num_golden_per_class.keys()}

    # (4) macro averaged p, r, f1, strict match
    macro_precision = sum([p / len(list(precision_per_class.values())) for p in list(precision_per_class.values())])
    macro_recall = sum([r / len(list(recall_per_class.values())) for r in list(recall_per_class.values())])
    macro_f1 = sum([f / len(list(f1_per_class.values())) for f in list(f1_per_class.values())])


    precision_per_class = dict(sorted(precision_per_class.items(), key=lambda x: x[0]))  # sorted by label name, for convinent browser
    recall_per_class = dict(sorted(recall_per_class.items(), key=lambda x: x[0]))
    f1_per_class = dict(sorted(f1_per_class.items(), key=lambda x: x[0]))

    return precision, recall, f1, precision_wo_label, recall_wo_label, f1_wo_label, precision_per_class, recall_per_class, f1_per_class, macro_precision, macro_recall, macro_f1


class BasicMetric:
    def __init__(self, metric_name, dataset_name, higher_the_better=True, id2classmapper=None):
        self.metric_name = metric_name
        self.dataset_name = dataset_name
        self.id2classmapper = id2classmapper
        self.history_value = []
        # self.value = 0.0
        self.epoch = 0
        self.higher_the_better = higher_the_better
        self._value = -1e10 if self.higher_the_better else 1e10
        self.current_best_value = self._value
        self.current_best_epoch = self.epoch
        self._not_used = True

    @property
    def value(self):
        if self._not_used:
            return None
        return self._value

    def update(self, **kwargs):
        self._not_used = False
        new_value = self._calculate_metric(**kwargs)
        self.epoch += 1
        self._update_current_best(new_value)

        self._value = new_value
        self.history_value.append(self.value)

    def _update_current_best(self, new_value):
        if self.higher_the_better and self._higher_than(new_value, self.current_best_value):
            print(f"MAX {self.metric_name} on {self.dataset_name} set update at Epoch {self.epoch}")
            self.current_best_value = new_value
            self.current_best_epoch = self.epoch
        if not self.higher_the_better and not self._higher_than(new_value, self.current_best_value):
            print(f"MIN {self.metric_name} on {self.dataset_name} set update at Epoch {self.epoch}")
            self.current_best_value = new_value
            self.current_best_epoch = self.epoch

    @staticmethod
    def _calculate_metric(**kwargs):
        raise NotImplementedError

    @staticmethod
    def _higher_than(metric_value_1, metric_value_2):
        return metric_value_1 >= metric_value_2


class ListAccuracyMetric(BasicMetric):
    """
    calculate father node prediction accuracy
    """
    @staticmethod
    def _calculate_metric(goldens, preds, mask_id=-1):
        assert len(goldens) == len(preds)
        correct = 0
        total = 0
        for g, p in zip(goldens, preds):
            if g == mask_id:
                continue
            if g == p:
                correct += 1
            total += 1
        # print(goldens, preds)
        return correct / total * 100

class MaskedListAccuracyMetric(BasicMetric):
    """
    calculate father node prediction accuracy
    """
    @staticmethod
    def _calculate_metric(golden, pred, mask=None):
        assert len(golden) == len(pred)
        if mask:
            assert len(golden) == len(mask)
        else:
            mask = [1] * len(golden)
        correct = 0
        total = 0
        for g, p, m in zip(golden, pred, mask):
            if m == 0:
                continue
            if g == p:
                correct += 1
            total += 1
        # print(goldens, preds)
        return correct / total * 100

class MultipleConditionListAccuracyMetric(BasicMetric):
    """
    calculate father node prediction accuracy
    """
    @staticmethod
    def _calculate_metric(goldens, preds, mask=None):
        assert len(goldens) == len(preds)
        num_positions = len(goldens[0])
        for i, j in zip(goldens, preds):
            assert len(i) == num_positions
            assert len(j) == num_positions
        if mask:
            assert len(mask) == num_positions
        else:
            mask = [1] * num_positions

        correct = 0
        total = 0
        for i in range(num_positions):
            if mask[i] == 0:
                continue
            is_correct = True
            for golden, pred in zip(goldens, preds):
                if golden[i] != pred[i]:
                    is_correct = False
            if is_correct:
                correct += 1
            total += 1
        """for g, p in zip(goldens, preds):
            if g == mask_id:
                continue
            if g == p:
                correct += 1
            total += 1"""
        # print(goldens, preds)
        return correct / total * 100

class MultipleClassificationAndListAccuracyMetric(BasicMetric):
    """
        calculate father/brother node prediction accuracy attributed by each class
    """
    def __init__(self, metric_name, dataset_name, higher_the_better=True,):
        super().__init__(metric_name, dataset_name, higher_the_better=True)
        self._value = tuple([-1] * 12)
        self.current_best_value = self._value
        # self.

    @property
    def value(self):
        if self._not_used:
            return None
        # precision, recall, f1, precision_wo_label, recall_wo_label, f1_wo_label,
        # precision_per_class, recall_per_class, f1_per_class, macro_precision, macro_recall, macro_f1
        return {"micro_LAS_precision": self._value[0],
                "micro_LAS_recall": self._value[1],
                "micro_LAS_f1": self._value[2],
                "micro_UAS_precision": self._value[3],
                "micro_UAS_recall": self._value[4],
                "micro_UAS_f1": self._value[5],
                "precision_per_class": self._value[6],
                "recall_per_class": self._value[7],
                "f1_per_class": self._value[8],
                "macro_precision": self._value[9],
                "macro_recall": self._value[10],
                "macro_f1": self._value[11],
                }

    @staticmethod
    def _calculate_metric(parent_goldens, parent_preds, parent_relation_goldens, parent_relation_preds, id2classmapper, mask_id=-1,):
        assert len(parent_goldens) == len(parent_preds)
        assert len(parent_goldens) == len(parent_relation_goldens)
        assert len(parent_goldens) == len(parent_relation_preds)
        """vertix_correct_node = 0
        attribute_correct_node = 0
        total_edge = 0
        count_for_class = defaultdict()
        for g, p, gr, pr in zip(parent_goldens, parent_preds, parent_relation_goldens, parent_relation_preds):
            if g == mask_id:
                continue
            if g == p:
                vertix_correct_node += 1
                if gr == pr:
                    attribute_correct_node += 1
                    id2classmapper[gr] += 1
            total_edge += 1
        # print(goldens, preds)
        return correct / total * 100"""
        parent_relation_goldens = [id2classmapper[i]  if i in id2classmapper.keys() else i for i in parent_relation_goldens]
        parent_relation_preds = [id2classmapper[i]  if i in id2classmapper.keys() else i for i in parent_relation_preds]
        # precision, recall, f1, precision_wo_label, recall_wo_label, f1_wo_label, precision_per_class, recall_per_class,\
        # f1_per_class, macro_precision, macro_recall, macro_f1 \
        #     = classwise_prf(parent_goldens, parent_preds, parent_relation_goldens, parent_relation_preds)
        return classwise_prf(parent_goldens, parent_preds, parent_relation_goldens, parent_relation_preds)

    @staticmethod
    def _higher_than(metric_value_1, metric_value_2):
        # print(metric_value_1, metric_value_2)
        return metric_value_1[11] >= metric_value_2[11]

# TODO
class TreePathExactAccuracyMetric(BasicMetric):
    pass

# TODO
class GraphEdgeSetAccuracyMetric(BasicMetric):
    pass


class SubdiscoursePRFMetric(BasicMetric):

    def __init__(self, metric_name, dataset_name, higher_the_better=True):
        super().__init__(metric_name, dataset_name, higher_the_better=True)
        self._value = (0, 0, -1e10)
        self.current_best_value = self._value

    """def _update_current_best(self, new_value):
        # print(new_value)
        # print(self.value)
        if self.higher_the_better and new_value[2] > self.value[2]:
            print(f"MAX {self.metric_name} on {self.dataset_name} set update at Epoch {self.epoch}")
            self.current_best_value = self.value
            self.current_best_epoch = self.epoch
        if not self.higher_the_better and new_value[2] < self.value[2]:
            print(f"MIN {self.metric_name} on {self.dataset_name} set update at Epoch {self.epoch}")
            self.current_best_value = self.value
            self.current_best_epoch = self.epoch"""

    @property
    def value(self):
        if self._not_used:
            return None
        return {"subdiscourse_p": self._value[0], "subdiscourse_r": self._value[1], "subdiscourse_f1": self._value[2],}


    @staticmethod
    def _higher_than(metric_value_1, metric_value_2):
        return metric_value_1[2] >= metric_value_2[2]

    @staticmethod
    def _calculate_metric_for_one_instance(parent_goldens, parent_preds, previous_goldens, previous_preds):
        # print("Parent Golden:", parent_goldens)
        # print("Previous Golden:", previous_goldens)
        # print("Parent Predicted:", parent_preds)
        # print("Previous Predicted:", previous_preds)
        golden_subdiscourses = subdiscourse_parser(parent_goldens, previous_goldens)
        predict_subdiscourses = subdiscourse_parser(parent_preds, previous_preds)
        # print("Golden:", golden_subdiscourses)
        # print("Predicted:", predict_subdiscourses)
        return prf(golden_list=golden_subdiscourses, predict_list=predict_subdiscourses)

    @staticmethod
    def _calculate_metric(parent_goldens, parent_preds, previous_goldens, previous_preds, length_of_discourse):
        # golden_subdiscourses = subdiscourse_parser(parent_goldens, previous_goldens)
        # predict_subdiscourses = subdiscourse_parser(parent_preds, previous_preds)
        total_num_golden, total_num_predict, total_num_correct = 0, 0, 0

        start_id = 0
        for length in length_of_discourse:
            instance_metrics = SubdiscoursePRFMetric._calculate_metric_for_one_instance(
                parent_goldens[start_id:start_id + length],
                parent_preds[start_id:start_id + length],
                previous_goldens[start_id:start_id + length],
                previous_preds[start_id:start_id + length])
            instance_precision, instance_recall, instance_f1, instance_num_golden, instance_num_predict, instance_num_correct = instance_metrics
            total_num_golden += instance_num_golden
            total_num_predict += instance_num_predict
            total_num_correct += instance_num_correct
            start_id += length

        total_precision = safe_devide(total_num_correct, total_num_golden) * 100
        total_recall = safe_devide(total_num_correct, total_num_predict) * 100
        if total_num_golden == 0 and total_num_predict == 0:
            total_precision, total_recall = 1.0, 1.0
        total_f1 = safe_devide(2 * total_precision * total_recall, total_precision + total_recall)

        return total_precision, total_recall, total_f1
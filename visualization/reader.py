from typing import List
import os
import pandas as pd

INSTANCE_HEAD = "===================="
GOLDEN_PREFIX = "golden parents: "
PREDICT_PREFIX = "predict parents: "
GOLDEN_PREVIOUS_PREFIX = "golden previous: "
PREDICT_PREVIOUS_PREFIX = "predict previous: "

id_to_label = {0:"Continue", 1:"Break", 2:"Combine", 3:"NA"}


class WebPage:
    def __init__(self, uid: int, text_list: List[str] = None, parent_list: List[int] = None, golden: bool = False, previous_list: List[int] = None):
        self._uid = uid
        self._text_list = text_list
        self._parent_list = parent_list
        self._golden = golden
        self._previous_relation_list = previous_list

    def set_text_list(self, text_list: List[str]):
        self._text_list = text_list

    def set_parent_list(self, parent_list: List[str]):
        self._parent_list = parent_list

    def set_uid(self, uid: int):
        self._uid = uid

    def set_previous_relation(self, relation_list: List):
        self._previous_relation_list = relation_list

    @property
    def uid(self):
        return self._uid

    @property
    def golden(self):
        return self._golden

    @property
    def text_list(self):
        return self._text_list

    @property
    def parent_list(self):
        return self._parent_list

    @property
    def previous_list(self):
        return self._previous_relation_list


class LabelReader:
    def __init__(self):
        self.webpage_golden = dict()
        self.webpage_predict = dict()
        self.unsolved_uid_list = []

    def __iter__(self):
        for webpage in list(self.webpage_golden.values()) + list(self.webpage_predict.values()):
            yield webpage

    def _read_label_file(self, path):
        with open(path, "r", encoding="utf-8") as labelfile:
            for line in labelfile:
                if line.startswith(INSTANCE_HEAD):
                    uid = int(line.strip().split("\t")[1][7:-6])
                if line.startswith(GOLDEN_PREFIX):
                    golden_father_list = eval(line.strip()[len(GOLDEN_PREFIX):])
                    self.webpage_golden[uid] = WebPage(uid=uid, parent_list=golden_father_list, golden=True)
                    self.unsolved_uid_list.append(uid)
                if line.startswith(PREDICT_PREFIX):
                    predict_father_list = eval(line.strip()[len(PREDICT_PREFIX):])
                    self.webpage_predict[uid] = WebPage(uid=uid, parent_list=predict_father_list, golden=False)
                if line.startswith(GOLDEN_PREVIOUS_PREFIX):
                    golden_relation_list = eval(line.strip()[len(GOLDEN_PREVIOUS_PREFIX):])
                    self.webpage_golden[uid].set_previous_relation(golden_relation_list)
                if line.startswith(PREDICT_PREVIOUS_PREFIX):
                    predict_relation_list = eval(line.strip()[len(PREDICT_PREVIOUS_PREFIX):])
                    self.webpage_predict[uid].set_previous_relation(predict_relation_list)

    def _read_source_file(self, dir):
        solved_list = []
        for uid in self.unsolved_uid_list:
            try:
                dataframe = pd.read_table(os.path.join(dir, str(uid)+".txt"), skiprows=[0])
                text_list = list(dataframe["Content"])
                self.webpage_golden[uid].set_text_list(text_list)
                self.webpage_predict[uid].set_text_list(text_list)
                solved_list.append(uid)
            except:
                pass
        self.unsolved_uid_list = list(set(self.unsolved_uid_list) - set(solved_list))

    def read_webpages(self, label_file_path, source_file_dir):
        self._read_label_file(label_file_path)
        self._read_source_file(source_file_dir)


# class SourceReader:
#     def __init__(self):

if __name__ == '__main__':
    dataframe = pd.read_table(os.path.join("../webpage_annotation", str(6) + ".txt"), skiprows=[0])
    text_list = list(dataframe["Content"])
    print(text_list)
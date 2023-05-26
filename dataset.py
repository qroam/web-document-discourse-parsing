# -*- coding: utf-8 -*-
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class WebDataset(Dataset):
    def __init__(self, path="data"):
        self.path = path
        self.filename = os.listdir(self.path)
        self.dataset = []
        self._read_all()

    @staticmethod
    def _read(file, headline=0):
        keys = []
        features = {}
        for i, line in enumerate(file):
            if i < headline:
                continue
            if i == headline:
                keys = line.split("\t")
                for k in keys:
                    features[k] = []
            if i > headline and line!="":  # add condition line!=""
                values = line.split("\t")
                values = [v if v else "NA" for v in values]
                for j, v in enumerate(values):
                    features[keys[j]].append(v)
        return features

    def _read_all(self):
        for filename in self.filename:
            with open(os.path.join(self.path, filename), "r", encoding="utf-8") as file:
                text = file.read()
                instance = self._read(text.split("\n"))
                instance["id"] = filename
                self.dataset.append(instance)
                # print(instance)

    def __getitem__(self, idx):
        # with open(os.path.join(self.path, self.filename[idx])) as file:
        #     text = file.read()
        #     instance = self._read(text.strip().split("\n"))
        # return instance
        return self.dataset[idx]

    def __len__(self):
        return len(self.filename)

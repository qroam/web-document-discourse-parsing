import json
import re
import nltk
import os
import argparse


import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from processor import WebDataProcessor
from utils import father_id_to_previous_id, common_collate_fn

from module import BaseNodeEncoder

class BaseTrainEnv:
    @staticmethod
    def prepare_tokenizer(args):
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        )
        return tokenizer

    @staticmethod
    def prepare_model(args, tokenizer, data_processor):
        raise NotImplementedError

    @staticmethod
    def get_param_groups(args, model):
        # param_groups = [{'params': model.parameters(), 'lr': args.learning_rate}]
        # return param_groups
        """param_groups = [{'params': [param for name, param in model.named_parameters() if
                        name.split('.')[0] != 'pretrained_model'], 'lr': args.learning_rate}]

        param_groups.append({'params': filter(lambda p: p.requires_grad, model.pretrained_model.parameters()),
                             'lr': args.transformer_learning_rate})"""
        
        """transformer_parameters = model.paragraph_encoder.paragraph_encoder.parameters() if hasattr(model.paragraph_encoder, "paragraph_encoder") else model.paragraph_encoder.parameters()
        print(list(map(id, transformer_parameters)))
        print(list(map(id, model.paragraph_encoder.paragraph_encoder.parameters())))
        
        param_groups = [{"params": transformer_parameters,
         "lr": args.transformer_learning_rate}]  # paragraph_encoder
        param_groups += [{"params": filter(lambda p: id(p) not in list(map(id, transformer_parameters)),
                                           model.parameters()),
                          "lr": args.learning_rate}]
        return param_groups"""
        # transformer_parameters = model.paragraph_encoder.paragraph_encoder.parameters() if hasattr(model.paragraph_encoder, "paragraph_encoder") else model.paragraph_encoder.parameters()
        
        if args.no_text_information:
            param_groups = [
                {"params": model.parameters(), "lr": args.learning_rate}
            ]
            return param_groups
        # print("bool", bool(hasattr(model.paragraph_encoder, "paragraph_encoder")))
        
        transformer_params = list(map(id, model.paragraph_encoder.paragraph_encoder.parameters()))
        param_groups = [{"params": model.paragraph_encoder.paragraph_encoder.parameters(),
                         "lr": args.transformer_learning_rate}]  # paragraph_encoder
        param_groups += [{"params": filter(lambda p: id(p) not in transformer_params,
                                           model.parameters()),
                          "lr": args.learning_rate}]
        print(param_groups)
        return param_groups

    @staticmethod
    def prepare_argparser():
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def prepare_dataprocessor(args, tokenizer):
        processor = WebDataProcessor(args, tokenizer)
        return processor

    @staticmethod
    def get_train_collate_fn(data_processor=None):
        return common_collate_fn

    @staticmethod
    def get_test_collate_fn(data_processor=None):
        return common_collate_fn

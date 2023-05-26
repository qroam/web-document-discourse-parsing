import json
import re
import nltk
import os


import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from .processor import DAMTProcessor
from .model import ParsingNet
from .args import parser as damt_parser
from .utils import eval_collate_fn, train_collate_fn_new

from module import BaseNodeEncoder

from train_utils import BaseTrainEnv

class DAMTTrainEnv(BaseTrainEnv):
    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group('damt')
        group.add_argument('--max_edu_dist', type=int, default=999)  # √
        # parser.add_argument('--glove_embedding_size', type=int, default=100)  ### use transformers vocab directly
        
        group.add_argument('--path_hidden_size', type=int, default=512)  # √  ### 384
        group.add_argument('--hidden_size', type=int, default=768)  # ×
        group.add_argument('--num_layers', type=int, default=3)  # √
        group.add_argument('--num_heads', type=int, default=4)  # √
        group.add_argument('--dropout', type=float, default=0.1)  # √  ### 0.5
        group.add_argument('--attention_dropout_DCA', type=float, default=0.1)  # This parameter has not been used in source code
        group.add_argument('--speaker', action='store_true')  # This parameter has not been used in source code
        group.add_argument('--valid_dist', type=int, default=99)  # √  ### 10
        group.add_argument('--decoder_input_size', type=int, default=512)  # √  ### 384
        group.add_argument('--decoder_hidden_size', type=int, default=512)  # √  ### 384
        group.add_argument('--classes_label', type=int, default=17)  # ×
        group.add_argument('--transition_weight', type=int, default=1, help="transition loss weight in multi-task loss")  # √
        group.add_argument('--graph_weight', type=int, default=1, help="graph loss weight in multi-task loss")  # √
        group.add_argument('--add_norm', type=bool, default= True)  # √
        
        group.add_argument('--dagcn_embedding_dims', type=int, default=8, help="d prime in the paper")  # √
        group.add_argument('--dagcn_valid_dist', type=int, default=99, help="the embedding space dimension for distance embedding in distance-aware graph module")  # √ 
        
        group.add_argument('--split_hidden_size', type=int, default=64)  # √
        group.add_argument('--biaffine_hidden_size', type=int, default=128)  # √

        group.add_argument('--unified_previous_classifier', action="store_true",)  # 1/1


    @staticmethod
    def prepare_tokenizer(args):
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        )
        return tokenizer
        # TODO
        """glove_tokenizer_path = os.path.join(args.dataset_dir, 'tokenizer.pt')
        if args.remake_tokenizer:
            tokenizer = GloveTokenizer(args)
            torch.save(tokenizer, glove_tokenizer_path)
        tokenizer = torch.load(glove_tokenizer_path)
        # pretrained_embedding = tokenizer.emb
        return tokenizer"""

    @staticmethod
    def prepare_model(args, tokenizer, data_processor):
        config = AutoConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            # num_labels=args.num_class,
        )
        config.gradient_checkpointing = True

        node_encoder = BaseNodeEncoder(args, config, data_processor)

        # model = StudentModel(args, config)
        model = ParsingNet(args, config, node_encoder)

        if args.test_only:
            # load_path = os.path.join(args.model_path, f"checkpoint_{args.test_checkpoint_id}.pkl")
            load_path = args.test_checkpoint_dir
            print(f"Loading NN model pretrained checkpoint from {load_path} ...")
            model.load_state_dict(torch.load(load_path))
        model.to(args.device)
        return model

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
        print("bool", bool(hasattr(model.paragraph_encoder, "paragraph_encoder")))
        
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
        return damt_parser

    @staticmethod
    def prepare_dataprocessor(args, tokenizer):
        processor = DAMTProcessor(args, tokenizer)
        return processor

    @staticmethod
    def get_train_collate_fn(data_processor=None):
        # return data_processor.train_collate_fn
        return train_collate_fn_new

    @staticmethod
    def get_test_collate_fn(data_processor=None):
        # return data_processor.test_collate_fn
        return eval_collate_fn

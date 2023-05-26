import json
import re
import nltk
import os


import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from .processor import SSAProcessor
from .model import StudentModel, StudentModelPLM
from .args import parser as ssa_parser
from .utils import eval_collate_fn as ssa_collate_fn

from module import BaseNodeEncoder

from train_utils import BaseTrainEnv

class GloveTokenizer:
    def __init__(self, args):
        self.args = args
        self.glove_vocab = self.load_glove_embedding()
        self.fdist = self.load_corpus()
        self.word2idx, self.emb = self.corpus_vocab()
        self.pad_token_id = 0
        self.glove_vocab = None

    def load_glove_embedding(self):
        glove_vocab = {}
        with open(self.args.glove_vocab_path, 'rb') as file:
            print(file.read().decode('utf-8').split("\n")[:5])  # TypeError: a bytes-like object is required, not 'str'
            for line in file.readlines():
                line = line.split()
                glove_vocab[line[0]] = np.array(line[1:]).astype(np.float)
        return glove_vocab

    def encode(self, text, special_token=True):
        if special_token:
            return [self.word2idx['CLS']] + [self.word2idx[word] if word in self.word2idx else self.word2idx['UNK'] for
                                             word in
                                             self.tokenize(text)]
        else:
            return [self.word2idx[word] if word in self.word2idx else self.word2idx['UNK'] for word in
                    self.tokenize(text)]

    @staticmethod
    def convert_number_to_special_token(tokens):
        # number to special token
        for i, token in enumerate(tokens):
            if re.match("\d+", token):
                tokens[i] = "[num]"
        return tokens

    @staticmethod
    def tokenize(text):
        return GloveTokenizer.convert_number_to_special_token(nltk.word_tokenize(text.lower()))

    def load_corpus(self):
        corpus_words = []
        for corpus_file in (self.args.train_file, self.args.eval_file, self.args.test_file):
            with open(corpus_file, 'r')as file:
                dataset = json.load(file)
                for data in dataset:
                    for edu in data['edus']:
                        corpus_words += self.tokenize(edu['text'])
        fdist = nltk.FreqDist(corpus_words)
        fdist = sorted(fdist.items(), reverse=True, key=lambda x: x[1])
        vocab = []
        for i, word in enumerate(fdist):
            word = word[0]
            if i < self.args.max_vocab_size or word in self.glove_vocab:
                vocab.append(word)
        return vocab

    def corpus_vocab(self):
        word2idx = {'PAD': 0, 'UNK': 1, 'CLS': 2, 'EOS': 3}
        define_num = len(word2idx)
        emb = [np.zeros(self.args.glove_embedding_size)] * define_num
        for idx, word in enumerate(self.fdist):
            word2idx[word] = idx + define_num
            if word in self.glove_vocab:
                emb.append(self.glove_vocab[word])
            else:
                emb.append(np.zeros(self.args.glove_embedding_size))
        print('corpus size : {}'.format(idx + define_num + 1))
        return word2idx, emb


class SSATrainEnv(BaseTrainEnv):
    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group('ssa')
        group.add_argument('--max_edu_dist', type=int, default=999)

        # model
        # parser.add_argument('--glove_embedding_size', type=int, default=300)  # TODO: wordvec is current invalid
        
        # parser.add_argument('--hidden_size', type=int, default=256)
        # parser.add_argument('--path_hidden_size', type=int, default=128)
        group.add_argument('--hidden_size', type=int, default=768,
            help="since residual connection is used, this size should be equal with encoder hidden dim")
        group.add_argument('--path_hidden_size', type=int, default=512)  # 256
        
        group.add_argument('--num_layers', type=int, default=3)  # layer number of GNN
        group.add_argument('--num_heads', type=int, default=4)  # head number of GNN, only used in StructureAwareAttention as a parameter
        # parser.add_argument('--num_layers', type=int, default=1)  # layer number of GNN
        # parser.add_argument('--num_heads', type=int, default=1)  # head number of GNN, only used in StructureAwareAttention as a parameter
        
        group.add_argument('--dropout', type=float, default=0.1)  # 0.5
        
        # parser.add_argument('--speaker', action='store_true')
        group.add_argument('--valid_dist', type=int, default=99)  # only used in PathEmbedding  # 10

        # TODO: implementation
        group.add_argument('--task', type=str, default="student", choices=["teacher", "student", "distill"])
        group.add_argument('--classify_loss', action='store_true')
        group.add_argument('--classify_ratio', type=float, default=0.2)
        group.add_argument('--distill_ratio', type=float, default=3.)
        
        # parser.add_argument('--use_negative_loss', type=bool, default=False)
        group.add_argument('--use_negative_loss', action="store_true",)  # 12/15
        group.add_argument('--negative_loss_weight', type=float, default=0.2)  # 12/15


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
        model = StudentModelPLM(args, config, node_encoder)


        if args.test_only:
            # load_path = os.path.join(args.model_path, f"checkpoint_{args.test_checkpoint_id}.pkl")
            load_path = args.test_checkpoint_dir
            print(f"Loading NN model pretrained checkpoint from {load_path} ...")
            model.load_state_dict(torch.load(load_path))
        model.to(args.device)
        return model

    @staticmethod
    def prepare_argparser():
        return ssa_parser

    @staticmethod
    def prepare_dataprocessor(args, tokenizer):
        processor = SSAProcessor(args, tokenizer)
        return processor

    @staticmethod
    def get_train_collate_fn(data_processor=None):
        return ssa_collate_fn

    @staticmethod
    def get_test_collate_fn(data_processor=None):
        return SSATrainEnv.get_train_collate_fn()

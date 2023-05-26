# -*- coding: utf-8 -*-

import os 
from typing import List

import torch
from torch import nn
from collections import defaultdict

       

class BasicEmbeddingLayer(nn.Module):
    def __init__(self, vocab, embedding_dim, pretrained_embedding=None, freeze=False):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = vocab.size
        self.embedding_dim = embedding_dim
        
        if pretrained_embedding is None:
            embedding_saved_in_vocab = vocab.get_embeddings()
            if embedding_saved_in_vocab is not None:
                pretrained_embedding = embedding_saved_in_vocab
        
        if pretrained_embedding is not None:  # from pretrained
            self.embedding = nn.Embedding.from_pretrained(embeddings=pretrained_embedding, freeze=freeze)
        else:  # from scratch, random
            self.embedding = nn.Embedding(self.vocab.size, embedding_dim)

        print("initialized BasicEmbeddingLayer")

    
    def forward(self, input_ids):
        """
        :param input_ids: (Batch_size, *, int<self.vocab.size)
        """
        return self.embedding(input_ids)
    
    def save_embeddings(self, dir):
        self.vocab.load_weight(self.embedding.weight)
        self.vocab.save_embeddings(dir)

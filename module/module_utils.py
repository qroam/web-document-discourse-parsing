import torch
from torch import nn

from .paragraph_encoder.bert_encoder import BERTEncoder
from .paragraph_encoder.rnn_encoder import RNNEncoder
from .paragraph_encoder.sentence_encoders import SentenceEncoder, sentence_encoder_model_name_list
from .paragraph_encoder.concat_document_encoder import ConcatDocumentEncoder  # 12/26

from .embedding.embedding import BasicEmbeddingLayer

from .xpath_encoder.xpath_lstm import XPathLSTM
from .xpath_encoder.xpath_ffnn import XPathFFNN


def prepare_html_tag_embedding(args, vocab):
    #####use html tag and XPath features#####
    """if args.use_html_embedding:
        self.html_tag_embedding = BasicEmbeddingLayer(vocab=vocab, embedding_dim=args.html_embedding_dim)
        self.hidden_dim += self.html_tag_embedding.embedding_dim"""
    html_tag_embedding = BasicEmbeddingLayer(vocab=vocab, embedding_dim=args.html_embedding_dim)
    return html_tag_embedding
    

def prepare_xpath_encoder(args, vocab):
    #####use html tag and XPath features#####
    """if args.use_xpath_embedding:
        self.xpath_encoder = None  # args.xpath_embedding_dim
        self.hidden_dim += self.xpath_encoder.embedding_dim"""
    xpath_encoder = None
    if args.xpath_encoder_type == "rnn":
        xpath_encoder = XPathLSTM(vocab, model_type="lstm", node_embedding_dim=args.xpath_node_embedding_dim, index_embedding_dim=args.xpath_index_embedding_dim, max_index_value=args.max_xpath_index_value)
    elif args.xpath_encoder_type == "ffnn":
        xpath_encoder = XPathFFNN(vocab, node_embedding_dim=args.xpath_node_embedding_dim, index_embedding_dim=args.xpath_index_embedding_dim, max_index_value=args.max_xpath_index_value)
    
    return xpath_encoder


def prepare_plm_text_encoder(args, config):
    if args.text_encoder_type == "concate":  # 12/16
        text_encoder = ConcatDocumentEncoder(model_name_or_path=args.model_name_or_path, config=config)
        return text_encoder
    if args.model_name_or_path in sentence_encoder_model_name_list:  # update 11/29
        text_encoder = SentenceEncoder(model_name=args.model_name_or_path, config=config)
    else:
        text_encoder = BERTEncoder(config=config)
    return text_encoder

def prepare_position_embedding(args):
    # TODO
    return


"""def prepare_aggregation_method(base_node_encoder, fixed_final_hidden_dim = 768):
    aggregation_method = base_node_encoder.aggregation_method

    if aggregation_method == "concat":
        final_hidden_dim = base_node_encoder.hidden_dim + base_node_encoder.html_tag_embedding_dim + base_node_encoder.xpath_embedding_dim + base_node_encoder.posotion_embedding_dim
        projector = lambda x:x
        def do_aggregation(para_embeddings, html_embeddings, xpath_embeddings, position_embeddings):
            para_embeddings = torch.cat((para_embeddings, html_embeddings, xpath_embeddings, position_embeddings), dim=-1)
            para_embeddings = self.projector(para_embeddings)
            return para_embeddings
    
    elif aggregation_method == "concat-project":
        final_hidden_dim = fixed_final_hidden_dim  # TODO
        projector = nn.Linear(self.hidden_dim + self.html_tag_embedding_dim + self.xpath_embedding_dim + self.posotion_embedding_dim, self.final_hidden_dim)
        def do_aggregation(para_embeddings, html_embeddings, xpath_embeddings, position_embeddings):
            para_embeddings = torch.cat((para_embeddings, html_embeddings, xpath_embeddings, position_embeddings), dim=-1)
            para_embeddings = self.projector(para_embeddings)
            return para_embeddings

    elif aggregation_method == "add":
        final_hidden_dim = max(base_node_encoder.hidden_dim, base_node_encoder.html_tag_embedding_dim, base_node_encoder.xpath_embedding_dim, base_node_encoder.posotion_embedding_dim)


    elif aggregation_method == "linear-combination-single-weight":
        final_hidden_dim = max(base_node_encoder.hidden_dim, base_node_encoder.html_tag_embedding_dim, base_node_encoder.xpath_embedding_dim, base_node_encoder.posotion_embedding_dim)

    elif aggregation_method == "linear-combination-complex-weight":
        final_hidden_dim = max(base_node_encoder.hidden_dim, base_node_encoder.html_tag_embedding_dim, base_node_encoder.xpath_embedding_dim, base_node_encoder.posotion_embedding_dim)
        
    return final_hidden_dim, do_aggregation"""
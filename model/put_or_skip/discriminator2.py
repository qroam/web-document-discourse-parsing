from typing import Dict
from collections import defaultdict
import random

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModel, AutoTokenizer

# from .module.paragraph_encoder import BERTEncoder, LSTMEncoder, SentenceEncoder
from module import BERTEncoder, RNNEncoder, SentenceEncoder

from .putorskip_processor import get_rightmost_branch_position, get_rightmost_branch_plus_title_position, get_all_position, get_parent_to_children_dict
from .putorskip_processor import get_context_of_insert_position_new as get_context_of_insert_position  # 11/28
from .putorskip_processor import position_method

from module import PairClassifier, DocumentLevelRNNEncoder, BasicEmbeddingLayer, BaseNodeEncoder
from utils import predict_previous_relations, predict_previous_relations_head_and_tail

from module import ArbitraryPairClassifier

# position_method = {
#     "right": get_rightmost_branch_position,
#     "right+maintitle": get_rightmost_branch_plus_title_position,
#     "all": get_all_position,
# }

class LSTMIntegrator(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=512, out_dim=512, num_layers=1):
        super().__init__()
        self.context_lstm = nn.LSTM(
            input_size=in_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True,
            # proj_size=out_dim,
            # __init__() got an unexpected keyword argument 'proj_size'
        )
        self.hidden_dim = 2 * hidden_dim

    def forward(self, inputs, context_lengths):
        """

        :param inputs: (batch_size, num_context_sentence, in_dim=sentence_embedding_dim)
        :param context_lengths: List of size (batch_size,), indicating the padding index of each instance
        :return:
        """
        # output, (h_n, c_n) = self.context_lstm(inputs)  # output: (batch_size, num_token, out_dim*bidirectional)
        packed_inputs = nn.utils.rnn.pack_padded_sequence(input=inputs, lengths=context_lengths, batch_first=True, enforce_sorted=False)
        output, (h_n, c_n) = self.context_lstm(packed_inputs)
        # h_n containing the final hidden state for each element in the sequence
        # When bidirectional=True, h_n will contain a concatenation of the final forward and reverse hidden states, respectively.
        h_n_lastlayer_forward = h_n[-2,:,:]  # (, batch_size, hidden_dim)
        h_n_lastlayer_backward = h_n[-1,:,:]  # (, batch_size, hidden_dim)
        context_representation = torch.cat((h_n_lastlayer_forward, h_n_lastlayer_backward), dim=-1)  # (, batch_size, hidden_dim * 2)
        # return output
        return context_representation


def pack_training_tuples_for_one_document(training_tuple_ids, para_embeddings):
    """
    Avoid repetitive computation of same paragraph embedding vector,
    let the construction of training tuples implemented in this Module class, rather than in data processing step
    :param training_tuple_ids: [((0,1,2,3),1), ((0,1,3),0), ...]
    :param para_embeddings: (num_paragraphs, paragraph_embedding_dim)
    :return:
    """
    labels = []
    # print(training_tuple_ids)
    training_tuple_ids = training_tuple_ids[0]  # squeeze the batch dim
    max_context_length = max([len(context_ids) for context_ids, label in training_tuple_ids])
    context_lengths = []
    context_tensors = []
    relative_pisition_distances = []  # 1108
    for context_ids, label in training_tuple_ids:
        labels.append(label)
        context_lengths.append(len(context_ids))
        # context_tensor = para_embeddings[:,list(context_ids),:].squeeze(0)
        context_tensor = para_embeddings[:,[i+1 for i in list(context_ids)],:].squeeze(0)  # 1021
        padding = torch.zeros(max_context_length-len(context_ids), context_tensor.shape[1]).to(context_tensor.device)
        # print(para_embeddings.shape)
        # print(list(context_ids))
        # print(context_tensor.shape)
        context_tensor = torch.cat((context_tensor, padding), dim=0)
        context_tensors.append(context_tensor)
        relative_pisition_distances.append(context_ids[-1] - context_ids[0])  # 1108
    return context_tensors, context_lengths, labels, relative_pisition_distances


class PutOrskipModel(nn.Module):
    # def __init__(self, in_dim, hidden_dim=512, out_dim=2, para_encoder=None, context_integrator=None):
    def __init__(self, args, config, node_encoder=None, context_integrator=None,):
        super().__init__()
        # self._para_encoder =
        self.paragraph_encoder = node_encoder or BERTEncoder(config=config)

        self.hidden_dim = self.paragraph_encoder.hidden_dim

        self.context_integrator = context_integrator or LSTMIntegrator(self.hidden_dim, args.hidden_dim)

        """self.use_relative_position = args.pos_use_relative_position
        self.relative_position_encoding_dim = args.pos_relative_position_encoding_dim if self.use_relative_position else 0
        self.relative_posotion_embedding = nn.Embedding(args.max_paragraph_num, args.pos_relative_position_encoding_dim) if self.use_relative_position else None"""
        self.use_relative_position = args.use_relative_position_embedding
        self.relative_position_encoding_dim = args.relative_position_embedding_dim if self.use_relative_position else 0
        self.relative_posotion_embedding = nn.Embedding(args.max_paragraph_num, self.relative_position_encoding_dim) if self.use_relative_position else None

        # self.classifier = nn.Linear(self.context_integrator.hidden_dim, args.out_dim)
        """self.classifier = nn.Linear(self.context_integrator.hidden_dim + self.relative_position_encoding_dim, args.parent_relation_dims)"""
        """self.classifier = nn.Linear(self.context_integrator.hidden_dim + self.relative_position_encoding_dim, args.parent_relation_dims + 2)  # 12/13"""
        # 12/31 将self.classifier由Linear改为MLP
        self.final_hidden_dim = self.context_integrator.hidden_dim + self.relative_position_encoding_dim
        self.classifier = nn.Sequential(nn.Linear(self.final_hidden_dim, self.final_hidden_dim),
                                        nn.Tanh(),
                                        nn.Linear(self.final_hidden_dim, args.parent_relation_dims + 2),
                                        # nn.Tanh()
                                        )


        self.parent_relation_dims = args.parent_relation_dims
        self.loss = nn.CrossEntropyLoss()

        # self.dummy_node_embedding = nn.Embedding(1, self.paragraph_encoder.hidden_dim)
        # self.dummy_node_embedding = torch.randn(self.paragraph_encoder.hidden_dim, requires_grad=True).to(args.device)  # TODO: the tensor you manually created by torch's tensor creating method is on cpu by default
        self.dummy_node_embedding = torch.randn(self.hidden_dim, requires_grad=True).to(args.device)

        self.train_possible_position = args.train_possible_position
        self.decode_possible_position = args.decode_possible_position
        self.decode_position_method = position_method[args.decode_possible_position]

        self.negative_sampling_ratio = args.negative_sampling_ratio

        #####use previous task#####
        self.use_previous_joint_loss = args.use_previous_joint_loss
        self.previous_loss_ratio = args.previous_loss_ratio

        if args.use_previous_joint_loss:
            self.unified_previous_classifier = args.unified_previous_classifier
            if args.unified_previous_classifier:
                self.previous_relation_classifier = ArbitraryPairClassifier(
                    head_input_dims=self.hidden_dim,
                    tail_input_dims=self.hidden_dim,
                    output_dims=args.previous_relation_dims,
                    pair_interation_method=args.pair_interation_method_previous_label or "concate-mlp",
                    position_highlight_mlp=True,
                    # relative_position_embedding=self.relative_position_embedding if args.use_relative_position_embedding else None
                )  # 1/1
            else:
                self.previous_relation_classifier = PairClassifier(input_dims=2*self.hidden_dim, output_dims=args.previous_relation_dims, position_highlight_mlp=True)

        """if args.use_previous_joint_loss:
            # self.previous_relation_classifier = PairClassifier(input_dims=2 * self.paragraph_encoder.hidden_dim, combine_before=args.combine_before)
            # self.previous_relation_classifier = PairClassifier(input_dims=2 * self.hidden_dim, combine_before=args.combine_before)
            self.previous_relation_classifier = PairClassifier(input_dims=2 * self.hidden_dim, output_dims=args.previous_relation_dims, position_highlight_mlp=True)"""
        #####use previous task#####
        

    # def forward(self, inputs:Dict, training_tuple_ids):
    def predict(self, **kwargs):
        
        # return self.forward(kwargs["input_ids"], kwargs["input_mask"], kwargs["htmltag_ids"],)
        kwargs.pop("training_tuple_ids")
        kwargs.pop("golden_previous_ids")
        kwargs.pop("golden_previous_labels")
        # input_ids = kwargs.pop("input_ids")
        return self.forward(**kwargs)

    def forward(self, input_ids, input_mask=None, training_tuple_ids=None, golden_previous_ids=None, golden_previous_labels=None, **kwargs):
        """

        :param inputs: (batch_size, num_para, num_token, embedding_dim)
        :param mask: (batch_size, num_para, num_token)
        :return:
        """
        # para_embeddings = self.para_encoder(inputs=inputs, mask=mask)
        # para_embeddings = self.para_encoder(inputs)
        # para_embeddings = self.paragraph_encoder(input_ids, input_mask)
        para_embeddings = self.paragraph_encoder(input_ids, input_mask, **kwargs)
        
        
        #####use html tag and XPath features#####
        # para_embeddings = self._concate_local_features(para_embeddings, kwargs)
        #####use html tag and XPath features#####    
        # if self.use_global_encoder:  # 1108
        #     para_embeddings = para_embeddings + self.global_encoder(para_embeddings, num_nodes=[para_embeddings.shape[1]])  # Not a batch implementation
        
        para_embeddings = torch.cat((self.dummy_node_embedding.unsqueeze(0).unsqueeze(0), para_embeddings), dim=1)

        if training_tuple_ids is not None:
            if self.negative_sampling_ratio < 1.0:
                for i, batch_training_tuple in enumerate(training_tuple_ids):
                    # batch_training_tuple: List[Tuple]
                    """training_tuple_ids[i] = [t for t in batch_training_tuple if t[1]!=5 or random.random() < self.negative_sampling_ratio]  # 1106 negative sampling"""
                    training_tuple_ids[i] = [t for t in batch_training_tuple if t[1]!=self.parent_relation_dims+1 or random.random() < self.negative_sampling_ratio]
            # training stage
            context_tensors, context_lengths, labels, relative_pisition_distances = pack_training_tuples_for_one_document(training_tuple_ids, para_embeddings)
            # print(context_tensors)
            # context_tensors = torch.tensor(context_tensors, dtype=torch.float32).to(para_embeddings.device)
            context_tensors = torch.stack(context_tensors, dim=0).to(para_embeddings.device)  # (context_batch_size, max_context_len, node_hidden_dim)
            labels = torch.tensor(labels, dtype=torch.long).to(para_embeddings.device)
            context_features = self.context_integrator(context_tensors, context_lengths)  # List, len = context_batch_size
            if self.use_relative_position:  # 1108
                context_features = torch.concat((context_features, self.relative_posotion_embedding(torch.tensor(relative_pisition_distances, dtype=torch.long).to(para_embeddings.device))), dim=-1)
            labels_predicted = self.classifier(context_features)

            # print(labels_predicted.shape)
            # print(labels)
            # print(labels.shape)

            loss = self.loss(labels_predicted, labels)
            outputs = {"loss": loss,}

        else:
            # inference stage
            outputs = self._decode(para_embeddings)
        
        # print(golden_previous_ids)
        # print(len(golden_previous_ids))
        # print(para_embeddings.shape)

        #####use previous task#####
        if self.use_previous_joint_loss:
            if self.unified_previous_classifier:  # 1/1
                outputs = predict_previous_relations_head_and_tail(
                    outputs,
                    classifier=self.previous_relation_classifier,
                    head_encodings=para_embeddings,
                    tail_encodings=para_embeddings,
                    predicted_father_ids=outputs["father_ids"] if "father_ids" in outputs.keys() else None,
                    golden_previous_ids=golden_previous_ids,
                    golden_previous_labels=golden_previous_labels,  # TODO
                    loss_ratio=self.previous_loss_ratio,
                )
            else:
                outputs = predict_previous_relations(
                    outputs,
                    classifier=self.previous_relation_classifier,
                    encodings=para_embeddings,
                    predicted_father_ids=outputs["father_ids"] if "father_ids" in outputs.keys() else None,
                    golden_previous_ids=golden_previous_ids,
                    golden_previous_labels=golden_previous_labels,  # TODO
                    loss_ratio=self.previous_loss_ratio,
                )
        #####use previous task#####
        return outputs
        

    def _decode(self, para_embeddings, beam_size=1):
        """
        implement beam search for decoding
        :param para_embeddings:
        :param beam_size:
        :return:
        """
        beam = []
        seed_father_ids = []
        seed_father_labels = []
        seed_score = 0.0
        # scores = 0.0
        # parent_to_child = defaultdict(list)
        # parent_to_child[-1] = []
        # beam.append(parent_to_child)
        beam.append({"father_ids": seed_father_ids, "father_labels": seed_father_labels, "score": seed_score})
        batch_size, num_paragraph, hidden_dim = para_embeddings.shape

        # for decode_step in range(para_embeddings.shape[0]):
        # for decode_step in range(1, num_paragraph):
        for decode_step in range(0, num_paragraph-1):
            new_beam = []
            for current_tree in beam:
                parent_to_child = get_parent_to_children_dict(current_tree["father_ids"])  # {-1: []}
                # possible_insertion_positions = get_all_position(parent_to_child)  # [-1]
                possible_insertion_positions = self.decode_position_method(parent_to_child)
                
                inference_context_ids = []  # [((0,1,2,3),1), ((0,1,3),0), ...]
                for position in possible_insertion_positions:  # -1
                    context_node_ids = get_context_of_insert_position(father_index=position, current_node_index=decode_step, parent_to_child=parent_to_child)
                    inference_context_ids.append((tuple(context_node_ids), 0))
                inference_context_ids = [inference_context_ids]  # add the beam dimension for unification
                context_tensors, context_lengths, _, relative_pisition_distances = pack_training_tuples_for_one_document(inference_context_ids, para_embeddings)
                # context_tensors = torch.tensor(context_tensors, dtype=torch.float32).to(para_embeddings.device)
                context_tensors = torch.stack(context_tensors, dim=0).to(para_embeddings.device)
                # labels = torch.tensor(labels, dtype=torch.int).to(para_embeddings.device)

                context_features = self.context_integrator(context_tensors, context_lengths)  # (num_possible_positions, hidden_dim)
                if self.use_relative_position:  # 1108
                    context_features = torch.concat((context_features, self.relative_posotion_embedding(torch.tensor(relative_pisition_distances, dtype=torch.long).to(para_embeddings.device))), dim=-1)
                labels_predicted = self.classifier(context_features)  # (num_possible_positions, out_dim)
                possibility = torch.softmax(labels_predicted, dim=-1)
                """y = torch.argmax(possibility[:, :-1], dim=-1).tolist()
                p_y = torch.max(possibility[:, :-1], dim=-1).values.tolist()""" 

                y = []
                p_y = []
                for i, position in enumerate(possible_insertion_positions):  # 12/13
                    if position == -1:
                        # y.append(self.parent_relation_dims)  # index of "NA" label
                        y.append(-1)  # index of "NA" label should be -1 in original FATHER_RELATION_dict!!!
                        p_y.append(possibility[i, -2])
                    else:
                        y.append(torch.argmax(possibility[i, :-2], dim=-1).tolist())
                        p_y.append(torch.max(possibility[i, :-2], dim=-1).values.tolist())


                for i, position in enumerate(possible_insertion_positions):
                    new_father_ids = current_tree["father_ids"] + [position]
                    new_father_labels = current_tree["father_labels"] + [y[i]]
                    new_score = current_tree["score"] + p_y[i]
                    new_beam.append({"father_ids": new_father_ids, "father_labels": new_father_labels, "score": new_score})

            beam = list(sorted(new_beam, key=lambda x: x["score"], reverse=True))[:beam_size]
            # print("====", beam)

        # previous_ids = []
        # previous_labels = []
        father_ids = beam[0]["father_ids"]
        father_labels = beam[0]["father_labels"]
        return {
            "father_ids": [father_ids],
            "father_labels": [father_labels],
            "previous_ids": None,
            "previous_labels": None,}

    def _decode_one_step(self, beam_size=1):
        pass






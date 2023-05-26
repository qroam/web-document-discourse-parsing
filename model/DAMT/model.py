import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .gcn import GraphConvolution, normalize_adj
from .utils import get_mask, compute_loss, record_eval_result
from .module import DAMT, Decoder, SplitAttention, BiaffineAttention, Classifier

from module import PairClassifier, GraphClassifier

from utils import predict_previous_relations, father_id_to_previous_id, predict_previous_relations_forward_with_graph_encodings
from module import BERTEncoder, RNNEncoder, SentenceEncoder

from module import ArbitraryPairClassifier, ArbitraryPairPointer


def get_distance_aware_adjacent_matrix_old(graphs, input_ids, splits_predict_, nr_score, dagcn_valid_dist=3):
    onehot_structure = torch.zeros(graphs.shape).float().to(input_ids.device)
    transition_structure = torch.zeros(graphs.shape).long()
    transition_rel = torch.zeros(graphs.shape).long()
    for ibatch, (link_score_, rel_score_) in enumerate(zip(splits_predict_, nr_score)):
        step = 2
        for link, rel in zip(link_score_.argmax(-1)[1:], rel_score_.argmax(-1)[1:]):
            #$# print(rel)
            if link != 0:  # why do not consider "NA" relation??
                onehot_structure[ibatch, step, link] = 1.0
                if abs(step-2-link.cpu().item())<=1: 
                    transition_structure[ibatch, step, link] = 1
                else:
                    transition_structure[ibatch, step, link] = 2
                """transition_structure[ibatch, step, link] = abs(step - link)  # 12/18"""
                transition_rel[ibatch, step, link] = rel[link]
            step += 1
    return onehot_structure, transition_structure, transition_rel


def get_distance_aware_adjacent_matrix_test_old(graphs, input_ids, link_list, rela_list, dagcn_valid_dist=3):
    onehot_structure = torch.zeros(graphs.shape).float().to(input_ids.device)  # (Batch_size, max_node_num, max_node_num)
    transition_structure = torch.zeros(graphs.shape).long()  # (Batch_size, max_node_num, max_node_num)
    transition_rel = torch.zeros(graphs.shape).long()  # (Batch_size, max_node_num, max_node_num)
    step = 2
    for link, rel in zip(link_list[1:], rela_list[1:]):
        if link != 0:
            onehot_structure[0, step, link] = 1.0
            if abs(step - 2 - link.item()) <= 1:
                transition_structure[0, step, link] = 1
            else:
                transition_structure[0, step, link] = 2
            """transition_structure[0, step, link] = abs(step - link)  # 12/18"""
            transition_rel[0, step, link] = rel
            step += 1
    return onehot_structure, transition_structure, transition_rel

def get_distance_aware_adjacent_matrix(graphs, input_ids, splits_predict_, nr_score, dagcn_valid_dist=3):
    onehot_structure = torch.zeros(graphs.shape).float().to(input_ids.device)
    transition_structure = torch.zeros(graphs.shape).long()
    transition_rel = torch.zeros(graphs.shape).long()
    for ibatch, (link_score_, rel_score_) in enumerate(zip(splits_predict_, nr_score)):
        step = 2
        for link, rel in zip(link_score_.argmax(-1)[1:], rel_score_.argmax(-1)[1:]):
            #$# print(rel)
            if link != 0:  # why do not consider "NA" relation??
                onehot_structure[ibatch, step, link] = 1.0
                transition_structure[ibatch, step, link] = min(abs(step-2-link.cpu().item()), dagcn_valid_dist-1)  # -1 because embedding index is from 0
                transition_rel[ibatch, step, link] = rel[link]
            step += 1
    return onehot_structure, transition_structure, transition_rel


def get_distance_aware_adjacent_matrix_test(graphs, input_ids, link_list, rela_list, dagcn_valid_dist=3):
    onehot_structure = torch.zeros(graphs.shape).float().to(input_ids.device)  # (Batch_size, max_node_num, max_node_num)
    transition_structure = torch.zeros(graphs.shape).long()  # (Batch_size, max_node_num, max_node_num)
    transition_rel = torch.zeros(graphs.shape).long()  # (Batch_size, max_node_num, max_node_num)
    step = 2
    for link, rel in zip(link_list[1:], rela_list[1:]):
        if link != 0:
            onehot_structure[0, step, link] = 1.0
            """if abs(step - 2 - link.item()) <= 1:
                transition_structure[0, step, link] = 1
            else:
                transition_structure[0, step, link] = 2"""
            """transition_structure[0, step, link] = abs(step - link)  # 12/18"""
            transition_structure[0, step, link] = min(abs(step-2-link), dagcn_valid_dist-1)  # -1 because embedding index is from 0
            transition_rel[0, step, link] = rel
            step += 1
    return onehot_structure, transition_structure, transition_rel


class ParsingNet(nn.Module):
    # def __init__(self, params, pretrained_model):
    def __init__(self, args, config, node_encoder=None,):
        super(ParsingNet, self).__init__()
        self.args = args

        # self.pretrained_model = pretrained_model
        self.paragraph_encoder = node_encoder or BERTEncoder(model_name_or_path=args.model_name_or_path, config=config)
        self.encoder_hidden_dim = self.paragraph_encoder.hidden_dim
        args.hidden_size = self.encoder_hidden_dim  # 12/10
        

        # self.encoder = DAMT(args, self.pretrained_model)
        self.encoder = DAMT(args, self.paragraph_encoder)
        self.decoder = Decoder(args.decoder_input_size, args.decoder_hidden_size)
        # self.splitAttention = SplitAttention(384, 384, 64)
        # self.nr_classifier = BiaffineAttention(384, 384, args.parent_relation_dims, 128)  # args.classes_label
        self.splitAttention = SplitAttention(args.path_hidden_size, args.decoder_hidden_size, args.split_hidden_size)#64)
        self.nr_classifier = BiaffineAttention(args.path_hidden_size, args.decoder_hidden_size, args.parent_relation_dims+1, args.biaffine_hidden_size)# 128)  # args.classes_label  # 12/18: +1

        self.activation = torch.nn.ReLU()
        self.layer_norm = nn.LayerNorm(normalized_shape=args.path_hidden_size, elementwise_affine=False)
        self.GCN1 = GraphConvolution(args.path_hidden_size, args.path_hidden_size, act_func=self.activation,
                                     dropout_rate=args.dropout)  # first gcn
        """self.link_classifier = Classifier(args.path_hidden_size * 2, args.path_hidden_size, 1)
        self.label_classifier = Classifier(args.path_hidden_size * 2, args.path_hidden_size,
                                           args.parent_relation_dims+1)  # args.classes_label  # 12/17: +1"""  # 1/1
        self.link_classifier = ArbitraryPairPointer(
            head_input_dims=args.path_hidden_size,
            tail_input_dims=args.path_hidden_size,
            output_dims=1,
            pair_interation_method=args.pair_interation_method_parent or "concate-mlp",
            position_highlight_mlp=False,
            # relative_position_embedding=self.relative_position_embedding if args.use_relative_position_embedding else None
        )  # 1/1

        self.label_classifier = ArbitraryPairClassifier(
            head_input_dims=args.path_hidden_size,
            tail_input_dims=args.path_hidden_size,
            output_dims=args.parent_relation_dims + 1,
            pair_interation_method=args.pair_interation_method_parent_label or "concate-mlp",
            position_highlight_mlp=False,
            # relative_position_embedding=self.relative_position_embedding if args.use_relative_position_embedding else None
        )  # 1/1


        # self.DisEmbedding = nn.Embedding(3, 8)  # TODO
        # self.RelEmbedding = nn.Embedding(17, 8)  # TODO
        self.DisEmbedding = nn.Embedding(args.dagcn_valid_dist, args.dagcn_embedding_dims)  # TODO
        self.RelEmbedding = nn.Embedding(args.parent_relation_dims+1, args.dagcn_embedding_dims)  # TODO  # 12/18: +1
        self.dagcn_valid_dist = args.dagcn_valid_dist
        # self.DisRelLinear = nn.Linear(16, 1)  # TODO
        self.DisRelLinear = nn.Linear(2*args.dagcn_embedding_dims, 1)  # TODO

        self.transition_weight = args.transition_weight
        self.graph_weight = args.graph_weight

        # self.use_negative_loss = args.use_negative_loss
        # self.negative_loss_weight = args.negative_loss_weight

        #####use previous task#####
        self.use_previous_joint_loss = args.use_previous_joint_loss
        self.previous_loss_ratio = args.previous_loss_ratio
        if args.use_previous_joint_loss:
            # self.previous_relation_classifier = PairClassifier(input_dims=args.path_hidden_size * 2, combine_before=args.combine_before)
            # self.previous_relation_classifier = GraphClassifier(input_dims=args.path_hidden_size * 2, combine_before=args.combine_before)
            """self.previous_relation_classifier = GraphClassifier(input_dims=args.path_hidden_size * 2, output_dims=args.previous_relation_dims)"""

            self.unified_previous_classifier = args.unified_previous_classifier  # 1/1
            if args.unified_previous_classifier:
                self.previous_relation_classifier = ArbitraryPairClassifier(
                    head_input_dims=args.path_hidden_size,
                    tail_input_dims=args.path_hidden_size,
                    output_dims=args.previous_relation_dims,
                    pair_interation_method=args.pair_interation_method_previous_label or "concate-mlp",
                    position_highlight_mlp=False,
                    # relative_position_embedding=self.relative_position_embedding if args.use_relative_position_embedding else None
                )  # 1/1
            else:
                self.previous_relation_classifier = GraphClassifier(input_dims=args.path_hidden_size * 2, output_dims=args.previous_relation_dims)
        #####use previous task#####

    def forward(self,
                # input_sentence,
                # sep_index_list,
                input_ids,
                input_mask,
                lengths,  # √
                edu_nums,  # √
                decoder_input,  # √
                decoder_mask,  # √
                speakers=None,  # √
                turns=None,  # √
                # d_outputs=None,  # √
                # d_output_re=None,  # √
                splits_ground=None,  # √, d_outputs in dataprocessor, splits_loss
                nrs_ground=None,  # √, d_output_re in dataprocessor, nr_loss
                graphs=None, 
                golden_previous_ids=None,
                golden_previous_labels=None,
                **kwargs):
        #$# print("forward()")
        """
        decoder_input
        decoder_mask
        splits_ground: (Batch, num_edu), gound father ids, no dummy node in length, father index from 0 (0 is dummy node), ONLY used in compute splits_loss
        nrs_ground
        graphs
        """
        
        """EncoderOutputs, Last_Hiddenstates, graph_predict_path, GEncoderoutputs = self.encoder(input_sentence, sep_index_list, lengths, edu_nums, speakers, turns, **kwargs)"""
        EncoderOutputs, Last_Hiddenstates, graph_predict_path, GEncoderoutputs = self.encoder(input_ids, input_mask, lengths, edu_nums, speakers, turns, **kwargs)
        Last_Hiddenstates = Last_Hiddenstates.squeeze(0)  # (1, batch, gru_hidden_dim) -> (batch, gru_hidden_dim)
        
        # UniGRU module
        d_outputs, d_outputs_masks, d_masks = self._decode_batch(EncoderOutputs, Last_Hiddenstates, decoder_input, decoder_mask)
        # splits_ground, nrs_ground = grounds  # grounds is the golden label

        # splits_ground = d_outputs  # (2*hidden_size)
        # nrs_ground = d_output_re

        # Biaffine classifier between H_tm and h_dk to predict link and label for the transition-based module
        splits_attn = self.splitAttention(EncoderOutputs, d_outputs, d_masks)  # (Batch, num_edu, num_edu+1) torch.Size([1, 36, 37])
        #$# print("splits_attn :self.splitAttention(EncoderOutputs, d_outputs, d_masks)", splits_attn, splits_attn.shape)
        #$# print("splits_ground (INPUT)", splits_ground, splits_ground.shape)  # (Batch, num_edu) torch.Size([1, 36]), no dummy node in length, father index from 0 (0 is dummy node)
        splits_predict_ = splits_attn.log_softmax(dim=2)
        splits_ground_ = splits_ground.view(-1)
        splits_predict = splits_predict_.view(splits_ground_.size(0), -1)
        splits_masks = d_outputs_masks.view(-1).float()
        #$# print("splits_attn -> .log_softmax(dim=2) -> .view()")
        #$# print("splits_predict (input into F.nll_loss)", splits_predict, splits_predict.shape)  # torch.Size([36, 37])
        #$# print("splits_ground_ (input into F.nll_loss)", splits_ground_, splits_ground_.shape)  # torch.Size([36])
        splits_loss = F.nll_loss(splits_predict, splits_ground_, reduction="none")  # torch.Size([36])
        #$# print("splits_loss", splits_loss, splits_loss.shape)
        #$# print("splits_masks : splits_masks = d_outputs_masks.view(-1).float()", splits_masks, splits_masks.shape)  # mask for Batch padding, torch.Size([36])
        splits_loss = (splits_loss * splits_masks).sum() / splits_masks.sum()
        #$# print("splits_loss: splits_loss = (splits_loss * splits_masks).sum() / splits_masks.sum()", splits_loss)
        
        #$# print("="*20)
        nr_score = self.nr_classifier(EncoderOutputs, d_outputs)  # (Batch, num_edu, num_edu+1, num_class) torch.Size([1, 36, 37, 7])
        #$# print("nr_score :self.nr_classifier(EncoderOutputs, d_outputs)", nr_score, nr_score.shape)
        nr_score = nr_score.log_softmax(dim=-1) * d_masks.unsqueeze(-1).float()  # (Batch_size, num_relation, num_edu, num_edu)
        #$# print("nr_score :nr_score.log_softmax(dim=-1) * d_masks.unsqueeze(-1).float()", nr_score, nr_score.shape)

        
        '''onehot_structure = torch.zeros(graphs.shape).float().to(input_ids.device)
        transition_structure = torch.zeros(graphs.shape).long()
        transition_rel = torch.zeros(graphs.shape).long()
        #$# print(transition_rel.shape)
        for ibatch, (link_score_, rel_score_) in enumerate(zip(splits_predict_, nr_score)):
            step = 2
            for link, rel in zip(link_score_.argmax(-1)[1:], rel_score_.argmax(-1)[1:]):
                #$# print(rel)
                if link != 0:  # why do not consider "NA" relation??
                    onehot_structure[ibatch, step, link] = 1.0
                    if abs(step-2-link.cpu().item())<=1:
                        transition_structure[ibatch, step, link] = 1
                    else:
                        transition_structure[ibatch, step, link] = 2
                    """transition_structure[ibatch, step, link] = abs(step - link)  # 12/18"""
                    transition_rel[ibatch, step, link] = rel[link]
                step += 1'''
        
        onehot_structure, transition_structure, transition_rel = get_distance_aware_adjacent_matrix(graphs, input_ids, splits_predict_, nr_score)
        
        # To initialize the adjacent matrix for DAGCN, A' in the paper, through the structure predicted by the transition-based module
        #$# print("transition_structure", transition_structure, transition_structure.shape,)
        #$# print("transition_rel", transition_rel, transition_rel.shape, transition_rel.max())
        structure_embedding = self.DisEmbedding(transition_structure.to(input_ids.device))
        rel_embedding = self.RelEmbedding(transition_rel.to(input_ids.device))
        alpha = self.DisRelLinear(torch.cat((structure_embedding, rel_embedding),dim=-1)).squeeze(-1)
        weight_matrix = alpha.matmul(onehot_structure)

        nr_score = nr_score.view(nr_score.size(0) * nr_score.size(1), nr_score.size(2), nr_score.size(3))  # torch.Size([36, 37, 7])
        #$# print("nr_score :nr_score.view(nr_score.size(0) * nr_score.size(1), nr_score.size(2), nr_score.size(3))", nr_score.shape)
        target_nr_score = nr_score[torch.arange(nr_score.size(0)), splits_ground_]  ## if golden father_ids are given, use them to train the relation classifier module
        #$# print("nrs_ground (INPUT)", nrs_ground, nrs_ground.shape)  # (Batch, num_edu) torch.Size([1, 36]), no dummy node in length, label index from 0 (0 is "NA")
        target_nr_ground = nrs_ground.view(-1)
        #$# print("target_nr_score = nr_score[torch.arange(nr_score.size(0)), splits_ground_]; APPLY MASK using splits_ground_", target_nr_score.shape)
        #$# print("target_nr_ground = nrs_ground.view(-1) (input into F.nll_loss)", target_nr_ground.shape)
        nr_loss = F.nll_loss(target_nr_score, target_nr_ground)
        #$# print("nr_loss", nr_loss, nr_loss.shape)


        # calculation of GCN module
        gcn1_output = self.GCN1(GEncoderoutputs, weight_matrix)
        gcn_output = gcn1_output
        batch_size, node_num, hidden_size = gcn_output.size()
        gcn_output = gcn_output.unsqueeze(1).expand(batch_size, node_num, node_num, hidden_size)
        """gcn_output = torch.cat((gcn_output, gcn_output.transpose(1, 2)), dim=-1)"""  # 1/1
        graph_predict_path = graph_predict_path + gcn_output  # Residual connection

        # classifier based on the edge vectors output by GCN
        """link_scores = self.link_classifier(graph_predict_path).squeeze(-1)  # graph_predict_path: (batch_size, node_num, node_num, hidden_size)
        label_scores = self.label_classifier(graph_predict_path)"""  # 1/1
        link_scores = self.link_classifier.forward_with_graph_encodings_return_logits(graph_predict_path)  # (batch, max_node_num, max_node_num)  # 1/1
        label_scores = self.label_classifier.forward_with_graph_encodings_return_logits(graph_predict_path)

        mask = get_mask(node_num=edu_nums + 1, max_edu_dist= self.args.max_edu_dist).to(input_ids.device)

        link_loss, label_loss, _ = compute_loss(link_scores.clone(), label_scores.clone(), graphs, mask,
                                                            negative=False)  # graphs is the golden label
        
        graph_link_loss = link_loss.mean()
        graph_label_loss = label_loss.mean()

        outputs = {
            "graph_link_loss": graph_link_loss,
            "graph_label_loss": graph_label_loss,
            "split_loss": splits_loss,
            "rel_loss": nr_loss
        }

        loss = self.transition_weight * (splits_loss + nr_loss) + self.graph_weight * (graph_link_loss + graph_label_loss)
        outputs["loss"] = loss

        # TODO
        #####use previous task#####
        if self.use_previous_joint_loss:
            # outputs = self._get_predicted_values(outputs, edu_nums)
            """outputs = predict_previous_relations(
                outputs,
                classifier=self.previous_relation_classifier,
                encodings=graph_predict_path,  # (batch, max_node_num, max_node_num, 2 * params.path_hidden_size)
                # predicted_father_ids=outputs["father_ids"] if "father_ids" in outputs.keys() else None,
                golden_previous_ids=golden_previous_ids,
                golden_previous_labels=golden_previous_labels,
                loss_ratio=self.previous_loss_ratio,
            )"""  # 1/1

            if self.unified_previous_classifier:  # 1/1
                outputs = predict_previous_relations_forward_with_graph_encodings(
                    outputs,
                    classifier=self.previous_relation_classifier,
                    encodings=graph_predict_path,  # (batch, max_node_num, max_node_num, 2 * params.path_hidden_size)
                    # predicted_father_ids=outputs["father_ids"] if "father_ids" in outputs.keys() else None,
                    golden_previous_ids=golden_previous_ids,
                    golden_previous_labels=golden_previous_labels,  # TODO
                    loss_ratio=self.previous_loss_ratio,
                )
            else:
                outputs = predict_previous_relations(
                    outputs,
                    classifier=self.previous_relation_classifier,
                    encodings=graph_predict_path,  # (batch, max_node_num, max_node_num, 2 * params.path_hidden_size)
                    # predicted_father_ids=outputs["father_ids"] if "father_ids" in outputs.keys() else None,
                    golden_previous_ids=golden_previous_ids,
                    golden_previous_labels=golden_previous_labels,  # TODO
                    loss_ratio=self.previous_loss_ratio,
                )
        #####use previous task#####

        return outputs
    
    
    def predict(self, **kwargs):
        """eval_link_loss, eval_label_loss = self.eval_loss(texts,sep_index_list, graphs, lengths, \
                                                                      speakers, turns, edu_nums, pairs, \
                                                                      d_inputs, d_masks, d_outputs, d_output_re)
        link, rel, graph_predict_result = self.test(texts, sep_index_list, graphs, lengths, \
                                                                 speakers, turns, edu_nums, pairs, \
                                                                 d_inputs, d_masks, d_outputs, d_output_re)
        graph_link_loss, graph_label_loss, split_loss, rel_loss = \
                        model.Training_loss_batch(
                            texts,sep_index_list, graphs, lengths, speakers, turns, edu_nums, pairs, \
                            d_inputs, d_masks, grounds)"""

        # grounds = (d_outputs, d_output_re)
        # kwargs["graphs"]=None
        # kwargs["golden_previous_ids"]=None
        # kwargs["golden_previous_labels"]=None

        # eval_link_loss, eval_label_loss = self.eval_loss(**kwargs)
        outputs = self.test(**kwargs)
        return outputs


    def decode2(self, input, memory, state):
        d_input = memory[0,input].unsqueeze(0).unsqueeze(0)
        d_output, state = self.decoder(d_input, state)
        masks = torch.zeros(1, 1, memory.size(1), dtype=torch.uint8)
        masks[0, 0, :input+1] = 1
        masks = masks.to(input.device)
        split_scores = self.splitAttention(memory, d_output, masks)
        split_scores = split_scores.softmax(dim=-1)
        nr_score = self.nr_classifier(memory, d_output).softmax(dim=-1) * masks.unsqueeze(-1).float()
        split_scores = split_scores[0, 0].cpu().detach().numpy()
        nr_score = nr_score[0, 0].cpu().detach().numpy()
        return split_scores, nr_score, state

    def _decode_batch(self, e_outputs, d_init_states, d_inputs, d_masks):
        """
        Called by self.forward(), UniGRU module
        """
        d_outputs_masks = (d_masks.sum(-1) > 0).type_as(d_masks)
        d_inputs = e_outputs[torch.arange(e_outputs.size(0)), d_inputs.permute(1, 0)].permute( 1, 0, 2)
        d_inputs = d_inputs * d_outputs_masks.unsqueeze(-1).float()
        d_outputs = self.decoder.run_batch(d_inputs, d_init_states, d_outputs_masks)
        return d_outputs, d_outputs_masks, d_masks


    # def eval_loss(self, input_sentence, sep_index_list, lengths, edu_nums,
    #          decoder_input, graphs, speakers=None, turns=None, **kwargs):
    def eval_loss(self, input_ids, input_mask, lengths, edu_nums,
             decoder_input, graphs, speakers=None, turns=None, **kwargs):

        """EncoderOutputs, Last_Hiddenstates, graph_predict_path, GEncoderoutputs = self.encoder(input_sentence, sep_index_list, lengths, edu_nums, speakers, turns)"""
        EncoderOutputs, Last_Hiddenstates, graph_predict_path, GEncoderoutputs = self.encoder(input_ids, input_mask, lengths, edu_nums, speakers, turns, **kwargs)
        Last_Hiddenstates = Last_Hiddenstates
        state = Last_Hiddenstates.detach()
        link_list = []
        rela_list = []
        for input in decoder_input[0]:
            link_scores, rela_scores, state = self.decode2(input, EncoderOutputs, state)
            predict_link = link_scores.argmax(-1)
            link_list.append(predict_link)
            if predict_link != input.item():
                predict_rela = rela_scores[predict_link].argmax(-1)
                rela_list.append(predict_rela)
            else:
                rela_list.append(0)
        """onehot_structure = torch.zeros(graphs.shape).float().to(input_ids.device)
        transition_structure = torch.zeros(graphs.shape).long()
        transition_rel = torch.zeros(graphs.shape).long()
        step = 2
        for link,  rel in zip(link_list[1:], rela_list[1:]):
            if link != 0:
                onehot_structure[0, step, link] = 1.0
                if abs(step-2-link.item())<=1:
                    transition_structure[0, step, link] = 1
                else:
                    transition_structure[0, step, link] = 2
                transition_rel[0, step, link] = rel
                step += 1"""
        onehot_structure, transition_structure, transition_rel = get_distance_aware_adjacent_matrix_test(graphs, input_ids, link_list, rela_list)

        structure_embedding = self.DisEmbedding(transition_structure.to(input_ids.device))
        rel_embedding = self.RelEmbedding(transition_rel.to(input_ids.device))
        alpha = self.DisRelLinear(torch.cat((structure_embedding, rel_embedding), dim=-1)).squeeze(-1)
        weight_matrix = alpha.matmul(onehot_structure).to(input_ids.device)

        gcn1_output = self.GCN1(GEncoderoutputs, weight_matrix)
        gcn_output = gcn1_output
        batch_size, node_num, hidden_size = gcn_output.size()
        gcn_output = gcn_output.unsqueeze(1).expand(batch_size, node_num, node_num, hidden_size)
        gcn_output = torch.cat((gcn_output, gcn_output.transpose(1, 2)), dim=-1)
        graph_predict_path = graph_predict_path + gcn_output

        graph_link_scores = self.link_classifier(graph_predict_path).squeeze(-1)
        graph_label_scores = self.label_classifier(graph_predict_path)
        mask = get_mask(node_num=edu_nums + 1, max_edu_dist= self.args.max_edu_dist).to(input_ids.device)
        eval_link_loss, eval_label_loss = compute_loss(graph_link_scores, graph_label_scores, graphs, mask)
        return eval_link_loss,eval_label_loss

    # def test(self, input_sentence, sep_index_list, lengths, edu_nums, pairs,
    #          decoder_input, graphs, speakers=None, turns=None, **kwargs):
    def test(self, input_ids, input_mask, lengths, edu_nums, pairs,
             decoder_input, graphs, speakers=None, turns=None, **kwargs):
        #$# print("test()")
        """
        return:
        link_list: father_ids predicted by transition module
        rela_list: father_labels predicted by transition module
        graph_predict_result: 
        """
        """EncoderOutputs, Last_Hiddenstates, graph_predict_path, GEncoderoutputs= self.encoder(input_sentence, sep_index_list, lengths, edu_nums, speakers, turns)"""
        EncoderOutputs, Last_Hiddenstates, graph_predict_path, GEncoderoutputs = self.encoder(input_ids, input_mask, lengths, edu_nums, speakers, turns, **kwargs)
        Last_Hiddenstates = Last_Hiddenstates
        state = Last_Hiddenstates.detach()

        link_list = []  # List[int]
        rela_list = []  # List[int]

        for input in decoder_input[0]:  ## alway set batch_size = 1 in test time
            link_scores, rela_scores, state = self.decode2(input, EncoderOutputs,state)
            predict_link = link_scores.argmax(-1)
            link_list.append(predict_link)
            if predict_link!=input.item():
                predict_rela = rela_scores[predict_link].argmax(-1)
                rela_list.append(predict_rela)
            else:
                rela_list.append(0)

        '''onehot_structure = torch.zeros(graphs.shape).float().to(input_ids.device)
        transition_structure = torch.zeros(graphs.shape).long()
        transition_rel = torch.zeros(graphs.shape).long()
        step = 2
        for link, rel in zip(link_list[1:], rela_list[1:]):
            if link != 0:
                onehot_structure[0, step, link] = 1.0
                if abs(step - 2 - link.item()) <= 1:
                    transition_structure[0, step, link] = 1
                else:
                    transition_structure[0, step, link] = 2
                """transition_structure[0, step, link] = abs(step - link)  # 12/18"""
                transition_rel[0, step, link] = rel
                step += 1'''
        onehot_structure, transition_structure, transition_rel = get_distance_aware_adjacent_matrix_test(graphs, input_ids, link_list, rela_list)

        structure_embedding = self.DisEmbedding(transition_structure.to(input_ids.device))
        rel_embedding = self.RelEmbedding(transition_rel.to(input_ids.device))
        alpha = self.DisRelLinear(torch.cat((structure_embedding, rel_embedding), dim=-1)).squeeze(-1)
        weight_matrix = alpha.matmul(onehot_structure).to(input_ids.device)

        gcn1_output = self.GCN1(GEncoderoutputs, weight_matrix)
        gcn_output = gcn1_output
        batch_size, node_num, hidden_size = gcn_output.size()
        gcn_output = gcn_output.unsqueeze(1).expand(batch_size, node_num, node_num, hidden_size)
        """gcn_output = torch.cat((gcn_output, gcn_output.transpose(1, 2)), dim=-1)""" 
        graph_predict_path = graph_predict_path + gcn_output
        """graph_link_scores = self.link_classifier(graph_predict_path).squeeze(-1)
        graph_label_scores = self.label_classifier(graph_predict_path)""" 
        graph_link_scores = self.link_classifier.forward_with_graph_encodings_return_logits(graph_predict_path)  # (batch, max_node_num, max_node_num) 
        graph_label_scores = self.label_classifier.forward_with_graph_encodings_return_logits(graph_predict_path)

        mask = get_mask(node_num=edu_nums + 1, max_edu_dist= self.args.max_edu_dist).to(input_ids.device)
        
        batch_size = graph_link_scores.size(0)
        max_len = edu_nums.max()
        graph_link_scores[~mask] = -1e9
        predicted_links = torch.argmax(graph_link_scores, dim=-1)
        predicted_labels = torch.argmax(graph_label_scores.reshape(-1, max_len + 1,  self.args.parent_relation_dims+1)[  # args.classes_label    #  +1
                                            torch.arange(batch_size * (max_len + 1)), predicted_links.reshape(
                                                -1)].reshape(batch_size, max_len + 1,  self.args.parent_relation_dims+1),  # args.classes_label    #  +1
                                        dim=-1)
        predicted_links = predicted_links[:, 1:] - 1
        predicted_labels = predicted_labels[:, 1:]
        predicted_labels = predicted_labels - 1 
        hp_pairs = {}
        step = 1
        while step < edu_nums[0]:
            link = predicted_links[0][step].item()
            label = predicted_labels[0][step].item()
            hp_pairs[(link, step)] = label  # link is the index of head node, step is the index of the current (tail) node
            step += 1
        graph_predict_result = {'hypothesis': hp_pairs,
                                  'reference': pairs[0],
                                  'edu_num': step}
        # return link_list, rela_list, graph_predict_result


        outputs = {
            # "eval_link_loss": eval_link_loss,
            # "eval_label_loss": eval_label_loss,
            "transition_based_father_ids": link_list,
            "transition_based_father_labels": rela_list,
            "graph_predict_result": graph_predict_result,
            "father_ids": predicted_links,
            "father_labels": predicted_labels,
            "previous_ids": None,
            "previous_labels": None,
        }

        #####use previous task#####
        if self.use_previous_joint_loss:
            # outputs = self._get_predicted_values(outputs, edu_nums)
            """outputs = predict_previous_relations(
                outputs,
                classifier=self.previous_relation_classifier,
                encodings=graph_predict_path,  # (batch, max_node_num, max_node_num, 2 * params.path_hidden_size)
                predicted_father_ids=outputs["father_ids"] if "father_ids" in outputs.keys() else None,
                # golden_previous_ids=golden_previous_ids,
                # golden_previous_labels=golden_previous_labels,
                loss_ratio=self.previous_loss_ratio,
            )"""  # 1/1

            if self.unified_previous_classifier:  # 1/1
                outputs = predict_previous_relations_forward_with_graph_encodings(
                    outputs,
                    classifier=self.previous_relation_classifier,
                    encodings=graph_predict_path,  # (batch, max_node_num, max_node_num, 2 * params.path_hidden_size)
                    predicted_father_ids=outputs["father_ids"] if "father_ids" in outputs.keys() else None,
                    # golden_previous_ids=golden_previous_ids,
                    # golden_previous_labels=golden_previous_labels,  # TODO
                    loss_ratio=self.previous_loss_ratio,
                )
            else:
                outputs = predict_previous_relations(
                    outputs,
                    classifier=self.previous_relation_classifier,
                    encodings=graph_predict_path,  # (batch, max_node_num, max_node_num, 2 * params.path_hidden_size)
                    predicted_father_ids=outputs["father_ids"] if "father_ids" in outputs.keys() else None,
                    # golden_previous_ids=golden_previous_ids,
                    # golden_previous_labels=golden_previous_labels,  # TODO
                    loss_ratio=self.previous_loss_ratio,
                )
        #####use previous task#####


        return outputs

# -*- coding: utf-8 -*-
import torch
from torch import nn

from module import BERTEncoder, RNNEncoder, SentenceEncoder
from module import ArbitraryPairClassifier, ArbitraryPairPointer
from module import PairClassifier 


from .module.self_pointer_network_new2 import PairwisePointerNetwork
from .module.pair_classifier import father_id_to_previous_id



class BaselineModel(nn.Module):
    # def __init__(self, args, config):
    def __init__(self, args, config, node_encoder=None,):
        super().__init__()
        
        self.paragraph_encoder = node_encoder or BERTEncoder(config=config)
        self.hidden_dim = self.paragraph_encoder.hidden_dim

        
        self.relative_position_embedding = nn.Embedding(args.max_paragraph_num, args.relative_position_embedding_dim) if args.use_relative_position_embedding else None
        
        self.father_net = ArbitraryPairPointer(
            head_input_dims=self.hidden_dim,
            tail_input_dims=self.hidden_dim,
            output_dims=1,
            pair_interation_method=args.pair_interation_method_parent,
            position_highlight_mlp=True,
            relative_position_embedding=self.relative_position_embedding if args.use_relative_position_embedding else None
        )
        # self.father_relation_classifier = ArbitraryPairClassifier(input_dims=2*self.hidden_dim, output_dims=args.parent_relation_dims)
        self.father_relation_classifier = ArbitraryPairClassifier(head_input_dims=self.hidden_dim,
                                                                  tail_input_dims=self.hidden_dim,
                                                                  output_dims=args.parent_relation_dims,
                                                                  pair_interation_method=args.pair_interation_method_parent_label,
                                                                  position_highlight_mlp=True,
                                                                  relative_position_embedding=self.relative_position_embedding if args.use_relative_position_embedding else None)  #  augmented pair classifier

        # print("initialized ArbitraryPairClassifier")

        self.relation_type_num = args.parent_relation_dims
        
        # self.previous_net = Pair_Classifier(input_dims=2 * self.paragraph_encoder.hidden_dim, combine_before=args.combine_before) if args.use_previous_joint_loss else None
        """self.previous_net = PairClassifier(input_dims=2 * self.paragraph_encoder.hidden_dim, output_dims=args.previous_relation_dims, position_highlight_mlp=True) if args.use_previous_joint_loss else None"""
        self.previous_net = ArbitraryPairClassifier(
            head_input_dims=self.hidden_dim,
            tail_input_dims=self.hidden_dim,
            output_dims=args.previous_relation_dims,
            pair_interation_method=args.pair_interation_method_previous_label,
            position_highlight_mlp=True,
            relative_position_embedding=self.relative_position_embedding if args.use_relative_position_embedding else None
        ) if args.use_previous_joint_loss else None



        self.use_previous_joint_loss = args.use_previous_joint_loss

        # self.dummy_node_embedding = nn.Embedding(1, self.paragraph_encoder.hidden_dim)
        self.dummy_node_embedding = torch.randn(self.hidden_dim, requires_grad=True).to(args.device)  # TODO: the tensor you manually created by torch's tensor creating method is on cpu by default

        # self.alpha = alpha
        self.alpha = args.alpha
        self.previous_loss_ratio = args.previous_loss_ratio

        print("initialized Baseline Model")
        # self.additional_encoder = args.additional_encoder  # Whether to use a Sequence_Modeling_Network layer

    def forward(self, input_ids, input_mask=None, golden_parent_ids=None, golden_parent_labels=None, golden_previous_ids=None, golden_previous_labels=None, **kwargs):
        encodings = self.paragraph_encoder(input_ids, input_mask, **kwargs)

        encodings = torch.cat((self.dummy_node_embedding.unsqueeze(0).unsqueeze(0), encodings), dim=1)
        """(father_node_logit_scores, father_loss) = self.father_net(encodings, golden_parent_ids)"""  # the returned father_node_logit_scores is without dummy node, while the father index for dummy node is 0
        (father_node_logit_scores, father_loss) = self.father_net(head_paragraphs=encodings, tail_paragraphs=encodings, golden_parent_ids=golden_parent_ids) 


        # predict previous relations
        if golden_parent_ids is None:  # inference
            fathers = torch.argmax(father_node_logit_scores, dim=2)
            fathers = fathers.tolist()
            # fathers = fathers - 1  # wipe off dummy node
        else:  # train, teacher forcing
            fathers = golden_parent_ids + 1

        previous_id_list = father_id_to_previous_id(fathers)  # father_id_to_previous_id() add the dummy node id 

        previous_id_list = torch.tensor(previous_id_list).to(input_ids.device)
        father_ids = torch.tensor(fathers).to(input_ids.device)  # (batch, num_nodes)
        father_ids = torch.cat((torch.zeros(father_ids.shape[0], 1).to(input_ids.device, dtype=father_ids.dtype), father_ids), dim=-1)  # add the dummy node id before input to father_relation_classifier
        (parent_relations_logit_scores, parent_relation_loss) = self.father_relation_classifier(head_paragraphs=encodings, tail_paragraphs=encodings, previous_ids=father_ids, golden=golden_parent_labels)
        
        
        if self.use_previous_joint_loss:
            """(previous_relations_logit_scores, previous_loss) = self.previous_net(encodings, previous_id_list, golden_previous_labels)"""
            (previous_relations_logit_scores, previous_loss) = self.previous_net(head_paragraphs=encodings, tail_paragraphs=encodings, previous_ids=previous_id_list, golden=golden_previous_labels)
        else:
            (previous_relations_logit_scores, previous_loss) = (None, None)

                    
        # fathers = fathers - 1  # wipe off dummy node
        if golden_parent_ids is not None:  # train
            if previous_loss is None:
                previous_loss = torch.tensor(0).to(input_ids.device)
            if parent_relation_loss is None: 
                parent_relation_loss = torch.tensor(0).to(input_ids.device)
            # loss = self.alpha * father_loss + (1 - self.alpha) * previous_loss
            """loss = (1-self.previous_loss_ratio) * (self.alpha * father_loss + (1 - self.alpha) * previous_loss) + self.previous_loss_ratio * previous_loss"""
            loss = (1-self.previous_loss_ratio) * (self.alpha * father_loss + (1 - self.alpha) * parent_relation_loss) + self.previous_loss_ratio * previous_loss

            return {
                "loss": loss,
                "father_loss": father_loss,
                "previous_loss": previous_loss,
                "father_ids": torch.argmax(father_node_logit_scores, dim=2) - 1,
                "previous_ids": previous_id_list[:, 1:] - 1,  # wipe off dummy node 
                "previous_label_logits": previous_relations_logit_scores,
                "previous_labels": torch.argmax(previous_relations_logit_scores, dim=-1) if previous_relations_logit_scores is not None else None,
                "father_relation_loss":parent_relation_loss,
                "father_labels": torch.argmax(parent_relations_logit_scores, dim=-1),
            }
        else:  # inference
            return {
                "father_ids": torch.argmax(father_node_logit_scores, dim=2) - 1,
                "previous_ids": previous_id_list[:, 1:] - 1,  # wipe off dummy node 
                "previous_label_logits": previous_relations_logit_scores,
                "previous_labels": torch.argmax(previous_relations_logit_scores, dim=-1) if previous_relations_logit_scores is not None else None,
                "father_labels": torch.argmax(parent_relations_logit_scores, dim=-1), 
            }
    
    
    def predict(self, **kwargs):
        kwargs["golden_parent_ids"]=None
        kwargs["golden_parent_labels"]=None
        kwargs["golden_previous_labels"]=None
        outputs = self.forward(**kwargs)
        # outputs.update(
        #     {
        #         "father_labels" : None,
        #     }
        # )
        return outputs
        


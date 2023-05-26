# -*- coding: utf-8 -*-
import torch
from torch import nn
from module import BERTEncoder, RNNEncoder, SentenceEncoder 
from module import ArbitraryPairClassifier, ArbitraryPairPointer
from module import PairClassifier 

from .module.pair_classifier import father_id_to_previous_id

from processor import FATHER_RELATION_dict


# reproduction of `Structured Global Representation` in DeepSeq, https://arxiv.org/pdf/1812.00176.pdf
class BaselineModelStructuredGlobal(nn.Module):
    # def __init__(self, args, config):
    def __init__(self, args, config, node_encoder=None,):
        super().__init__()
        
        self.paragraph_encoder = node_encoder or BERTEncoder(config=config)
        self.hidden_dim = self.paragraph_encoder.hidden_dim

        self.structured_representation_dim = args.structured_representation_dim
        self.relation_embedding_dim = args.relation_embedding_dim
        self.relation_embedding = nn.Embedding(len(FATHER_RELATION_dict.keys()), self.relation_embedding_dim)  # "NA" should also have embedding
        self.gru_cell = torch.nn.GRUCell(
            input_size=self.hidden_dim + self.relation_embedding_dim,
            hidden_size=self.structured_representation_dim,
        )

        # if args.additional_encoder:
        #     print("use contextual encoder")
        #     self.sequence_modeling_network = Sequence_Modeling_Network(d_model=self.paragraph_encoder.hidden_dim, model_type=args.additional_encoder_type)
        # deprecated 11/18

        self.father_net_head_mlp = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                                 nn.ReLU(),
                                                 nn.Linear(self.hidden_dim, self.hidden_dim),
                                                 nn.Tanh()
                                                 )
        self.father_net_tail_mlp = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                                 nn.ReLU(),
                                                 nn.Linear(self.hidden_dim, self.hidden_dim),
                                                 nn.Tanh()
                                                 )

        self.father_loss_type = args.loss_type
        self.father_loss = nn.MultiMarginLoss() if self.father_loss_type == "margin" else nn.CrossEntropyLoss()

        self.relative_position_embedding = nn.Embedding(args.max_paragraph_num, args.relative_position_embedding_dim) if args.use_relative_position_embedding else None

        self.father_net = ArbitraryPairPointer(
            head_input_dims=self.hidden_dim + self.structured_representation_dim,
            tail_input_dims=self.hidden_dim,
            output_dims=1,
            pair_interation_method=args.pair_interation_method_parent,
            position_highlight_mlp=True,
            relative_position_embedding=self.relative_position_embedding if args.use_relative_position_embedding else None
        )
        # self.father_relation_classifier = ArbitraryPairClassifier(input_dims=2*self.hidden_dim, output_dims=args.parent_relation_dims)
        self.father_relation_classifier = ArbitraryPairClassifier(
            head_input_dims=self.hidden_dim + self.structured_representation_dim,
            tail_input_dims=self.hidden_dim,
            output_dims=args.parent_relation_dims,
            pair_interation_method=args.pair_interation_method_parent_label,
            position_highlight_mlp=True,
            relative_position_embedding=self.relative_position_embedding if args.use_relative_position_embedding else None
        )  # 11/30, augmented pair classifier

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
        ) if args.use_previous_joint_loss else None  # 12/31
        self.use_previous_joint_loss = args.use_previous_joint_loss

        # self.dummy_node_embedding = nn.Embedding(1, self.paragraph_encoder.hidden_dim)
        self.dummy_node_embedding = torch.randn(self.hidden_dim, requires_grad=True).to(args.device)  # TODO: the tensor you manually created by torch's tensor creating method is on cpu by default

        # self.alpha = alpha
        self.alpha = args.alpha
        self.previous_loss_ratio = args.previous_loss_ratio
        # self.additional_encoder = args.additional_encoder  # Whether to use a Sequence_Modeling_Network layer

    def forward(self, input_ids, input_mask=None, golden_parent_ids=None, golden_parent_labels=None, golden_previous_ids=None, golden_previous_labels=None, **kwargs):

        encodings = self.paragraph_encoder(input_ids, input_mask, **kwargs)

        encodings = torch.cat((self.dummy_node_embedding.unsqueeze(0).unsqueeze(0), encodings), dim=1)

        batch_size, num_paragraphs, hidden_dim = encodings.shape

        head_vectors = self.father_net_head_mlp(encodings)  # (Batch, num_paragraph, hidden_dim)
        tail_vectors = self.father_net_tail_mlp(encodings)  # (Batch, num_paragraph, hidden_dim)


        father_node_logit_matrix = torch.zeros(batch_size, num_paragraphs-1, num_paragraphs).to(input_ids.device)
        father_relations_score_matrix = torch.zeros(batch_size, num_paragraphs-1, self.relation_type_num).to(input_ids.device)
        global_history_structure_matrix = torch.zeros(batch_size, num_paragraphs, self.structured_representation_dim).to(input_ids.device)  # the dummy root node should also have its global structrue vector
        parent_relation_loss = torch.tensor(0.).to(input_ids.device)
        
        hidden = torch.zeros(batch_size, self.structured_representation_dim).to(input_ids.device)


        if golden_parent_ids is not None:
            golden_parent_ids = golden_parent_ids + 1  # +1 to add the dummy node, otherwise it would be -1 !

        for t in range(num_paragraphs-1):
            # logfile.write("[incremental prediction] step_index (t)"+ str(t)+ "\n")
            # logfile.write("[incremental prediction] child idx in embedding matrix (t+1)"+ str(t+1)+ "\n")
            """father_node_logit_matrix[:,t,:] = self.father_net._pairwise_interaction_one_to_all(
                torch.cat((head_vectors, global_history_structure_matrix), dim=-1),
                tail_vectors,
                tail_index=t)"""
            father_node_logit_matrix[:,t,:] = self.father_net._pairwise_interaction_one_to_all(
                torch.cat((head_vectors, global_history_structure_matrix), dim=-1),
                tail_vectors,
                tail_index=t+1)  # t -> t+1

            """tail_embedding = encodings[:,t,:]"""
            tail_embedding = encodings[:,t+1,:]
            
            
            if golden_parent_ids is not None:  # teacher forcing
                golden_parent_index = golden_parent_ids[0][t]  # TODO batch implementation

                relative_position = (-torch.tensor(golden_parent_index)+t+1).unsqueeze(0).unsqueeze(0).to(input_ids.device)

                if golden_parent_labels is not None:
                    # logfile.write("[incremental prediction] current_step_golden_parent_labels (golden_parent_labels[:][t])" + str(golden_parent_labels[:,t].unsqueeze(1))+ "\n")
                    previous_relation_scores, loss_relation_t = self.father_relation_classifier.forward_without_dummy_node(  # TODO How to maintain relative position information here???
                        head_paragraphs=torch.cat((encodings[:,golden_parent_index,:].unsqueeze(1), global_history_structure_matrix[:,golden_parent_index,:].unsqueeze(1)), dim=-1),
                        tail_paragraphs=tail_embedding.unsqueeze(1),
                        golden=golden_parent_labels[:,t].unsqueeze(1),
                        relative_positions=relative_position
                    )
                    # predicted_relation_t = golden_parent_labels[0][t]  # TODO batch implementation
                    predicted_relation_t = golden_parent_labels[:,t]
                    parent_relation_loss += loss_relation_t
                else:
                    golden_parent_index = golden_parent_ids[0][t]  # TODO batch implementation
                    previous_relation_scores, loss_relation_t = self.father_relation_classifier.forward_without_dummy_node(
                        head_paragraphs=torch.cat((encodings[:,golden_parent_index,:].unsqueeze(1), global_history_structure_matrix[:,golden_parent_index,:].unsqueeze(1)), dim=-1),
                        tail_paragraphs=tail_embedding.unsqueeze(1),
                        relative_positions=relative_position
                    )
                    predicted_relation_t = torch.argmax(previous_relation_scores, dim=-1).squeeze(1)

            else:  # inference time
                # predicted_parent_index = torch.argmax(father_node_logit_matrix[0][t], dim=-1)  # TODO batch implementation
                predicted_parent_index = torch.argmax(father_node_logit_matrix[0][t][:t+1], dim=-1)  # TODO batch implementation
                relative_position = (-torch.tensor(predicted_parent_index)+(t+1)).unsqueeze(0).unsqueeze(0).to(input_ids.device)  # TODO batch implementation
                previous_relation_scores, loss_relation_t = self.father_relation_classifier.forward_without_dummy_node(
                        head_paragraphs=torch.cat((encodings[:,predicted_parent_index,:].unsqueeze(1), global_history_structure_matrix[:,predicted_parent_index,:].unsqueeze(1)), dim=-1),
                        tail_paragraphs=tail_embedding.unsqueeze(1),
                        relative_positions=relative_position
                    )
                predicted_relation_t = torch.argmax(previous_relation_scores, dim=-1).squeeze(1)
            
            father_relations_score_matrix[:,t,:] = previous_relation_scores
            
            # logfile.write("[incremental prediction] predicted_relation_t (golden when training)" + str(predicted_relation_t)+ "\n")
            if golden_parent_ids is not None:  # Here, if the predicted parent node is 0 (i.e. dummy node), then relation will be automatically assigned as "NA".
                predicted_relation_t = predicted_relation_t.masked_fill(golden_parent_index.unsqueeze(0)==0, value=-1)  # 0 = -1 + 1，stands for "NA" relation
            else:
                predicted_relation_t = predicted_relation_t.masked_fill(predicted_parent_index.unsqueeze(0)==0, value=-1)
            # logfile.write("[incremental prediction] predicted_relation_t_mask (input to relation embedding)" + str(predicted_relation_t)+ "\n")

            """relation_embedding = self.relation_embedding(predicted_relation_t) """
            relation_embedding = self.relation_embedding(predicted_relation_t + 1)  # +1 is because it is also required when the prediction relationship is "NA". That is to say, the space of relation_embedding is 1 more than the output dimension of the classifier
            # print(predicted_relation_t.shape, relation_embedding.shape, tail_embedding.shape)
            next_hidden = self.gru_cell(torch.cat((tail_embedding, relation_embedding), dim=-1), hidden)
            global_history_structure_matrix[:,t+1,:] = next_hidden
            hidden = next_hidden
                
            
        attn_mask = (torch.ones(num_paragraphs, num_paragraphs) - torch.tril(torch.ones(num_paragraphs, num_paragraphs))).to(input_ids.device)[1:,:]  # here, num_paragraphs is num edus + dummy node
        # father_node_logit_matrix = father_node_logit_matrix + (1 - attn_mask + (-10e9) * attn_mask)
        father_node_logit_matrix = father_node_logit_matrix + ((-10e9) * attn_mask)
        ## >>> >>> ##

        father_node_logit_scores = torch.softmax(father_node_logit_matrix, dim=2)
        # print(father_relations_score_matrix, father_relations_score_matrix.shape)
        parent_relations_logit_scores = father_relations_score_matrix
        # print(parent_relations_logit_scores, parent_relations_logit_scores.shape)

        # calculate father loss
        father_loss = None
        if golden_parent_ids is not None:
            # Compute loss
            flatten_golden = torch.flatten(golden_parent_ids)
            """flatten_golden = flatten_golden + 1  # Deal with "NA" type""" 
            flatten_father_node_logit_scores = torch.flatten(father_node_logit_scores, end_dim=1)
            flatten_father_node_logit_matrix = torch.flatten(father_node_logit_matrix, end_dim=1)  # TODO
            
            if self.father_loss_type == "ce":
                father_loss = self.father_loss(flatten_father_node_logit_matrix, flatten_golden)
            else:
                father_loss = self.father_loss(flatten_father_node_logit_scores, flatten_golden)
        
        father_relation_loss = None
        if golden_parent_labels is not None:
            father_relation_loss = parent_relation_loss / num_paragraphs


        # predict previous relations
        if golden_parent_ids is None:  # inference
            fathers = torch.argmax(father_node_logit_scores, dim=2)
            fathers = fathers.tolist()
            # fathers = fathers - 1  # wipe off dummy node
        else:  # train, teacher forcing
            """fathers = golden_parent_ids + 1"""
            fathers = golden_parent_ids
        # print(fathers)
        previous_id_list = father_id_to_previous_id(fathers)  # father_id_to_previous_id() add the dummy node id 
        # print(previous_id_list)
        previous_id_list = torch.tensor(previous_id_list).to(input_ids.device)
        
        # (previous_relations_logit_scores, previous_loss) = self.previous_net(encodings, previous_id_list, golden_previous_labels)
        if self.use_previous_joint_loss:
            """(previous_relations_logit_scores, previous_loss) = self.previous_net(encodings, previous_id_list, golden_previous_labels)"""
            (previous_relations_logit_scores, previous_loss) = self.previous_net(head_paragraphs=encodings, tail_paragraphs=encodings, previous_ids=previous_id_list, golden=golden_previous_labels)
        else:
            (previous_relations_logit_scores, previous_loss) = (None, None)

                    

        if golden_parent_ids is not None:  # train
            if previous_loss is None:
                previous_loss = torch.tensor(0).to(input_ids.device) 
            if parent_relation_loss is None: 
                father_relation_loss = torch.tensor(0).to(input_ids.device)
            # loss = self.alpha * father_loss + (1 - self.alpha) * previous_loss
            # loss = self.alpha * (father_loss + father_relation_loss) + (1 - self.alpha) * previous_loss
            """loss = (1-self.previous_loss_ratio) * (self.alpha * father_loss + (1 - self.alpha) * previous_loss) + self.previous_loss_ratio * previous_loss"""
            loss = (1-self.previous_loss_ratio) * (self.alpha * father_loss + (1 - self.alpha) * father_relation_loss) + self.previous_loss_ratio * previous_loss
            

            father_ids = torch.argmax(father_node_logit_scores, dim=2) - 1

            return {
                "loss": loss,
                "father_loss": father_loss,
                "previous_loss": previous_loss,
                "father_ids": father_ids,
                "previous_ids": previous_id_list[:, 1:] - 1,  # wipe off dummy node 
                "previous_label_logits": previous_relations_logit_scores,
                "previous_labels": torch.argmax(previous_relations_logit_scores, dim=-1) if previous_relations_logit_scores is not None else None,
                "father_relation_loss":father_relation_loss,
                "father_labels": torch.argmax(parent_relations_logit_scores, dim=-1).masked_fill(father_ids==-1, value=-1),
            }
        else:  # inference
            father_ids = torch.argmax(father_node_logit_scores, dim=2) - 1 
            # print(father_ids)
            # print(torch.argmax(parent_relations_logit_scores, dim=-1).masked_fill(father_ids==-1, value=-1))
            for i, j in zip(father_ids[0], torch.argmax(parent_relations_logit_scores, dim=-1).masked_fill(father_ids==-1, value=-1)[0]):
                assert j.tolist() in [-1, 0,1,2,3,4,5,6]
                assert i in list(range(-1, len(father_ids[0])))
                if i == -1:
                    assert j == -1
            return {
                "father_ids": father_ids,
                "previous_ids": previous_id_list[:, 1:] - 1,  # wipe off dummy node 
                "previous_label_logits": previous_relations_logit_scores,
                "previous_labels": torch.argmax(previous_relations_logit_scores, dim=-1) if previous_relations_logit_scores is not None else None,
                "father_labels": torch.argmax(parent_relations_logit_scores, dim=-1).masked_fill(father_ids==-1, value=-1), 
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
        


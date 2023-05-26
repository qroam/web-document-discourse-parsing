import torch
from torch import nn

from .utils import _get_clones
from .utils import *

from module import BERTEncoder
from module import DocumentLevelRNNEncoder

from module import PairClassifier, GraphClassifier
from utils import predict_previous_relations, father_id_to_previous_id, predict_previous_relations_forward_with_graph_encodings

from module import ArbitraryPairClassifier, ArbitraryPairPointer


class TeacherModel(nn.Module):
    def __init__(self, args, pretrained_embedding):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(embeddings=pretrained_embedding, freeze=False)
        self.sent_gru = nn.GRU(args.glove_embedding_size, args.hidden_size // 2, batch_first=True,
                               bidirectional=True)
        self.dialog_gru = nn.GRU(args.hidden_size, args.hidden_size // 2, batch_first=True, bidirectional=True)

        self.path_emb = PathEmbedding(args)
        self.path_model= PathModel(args)
        self.path_update = PathUpdateModel(args)
        self.gnn = StructureAwareAttention(args.hidden_size, args.path_hidden_size, args.num_heads, args.dropout)

        self.link_classifier = Classifier(args.path_hidden_size * 2, args.path_hidden_size, 1)
        self.label_classifier = Classifier(args.path_hidden_size * 2, args.path_hidden_size,
                                           args.parent_relation_dims)

        self.layer_num = args.num_layers
        self.norm = nn.LayerNorm(args.hidden_size)
        self.dropout = nn.Dropout(args.dropout)
        self.root = nn.Parameter(torch.zeros(args.hidden_size), requires_grad=False)

        self.hidden_size = args.hidden_size
        self.path_hidden_size = args.path_hidden_size


    def forward(self, texts, lengths, edu_nums, speakers, turns, graphs):
        batch_size, edu_num, sentence_len = texts.size()
        node_num = edu_num + 1

        texts = texts.reshape(batch_size * edu_num, sentence_len)
        texts = self.emb(texts)

        sent_output, sent_hx = self.sent_gru(texts)
        sent_output = self.dropout(sent_output)

        sent_output = sent_output.reshape(batch_size * edu_num, sentence_len, 2, -1)
        tmp = torch.arange(batch_size * edu_num)
        dialog_input = torch.cat((sent_output[tmp, lengths.reshape(-1) - 1, 0], sent_output[tmp, 0, 1]), dim=-1)
        dialog_input = torch.cat((self.root.expand(batch_size, 1, dialog_input.size(-1)),
                                  dialog_input.reshape(batch_size, edu_num, -1)),dim=1)

        dialog_output, dialog_hx = self.dialog_gru(dialog_input)
        dialog_output = self.dropout(dialog_output)

        node_nums = edu_nums + 1
        edu_attn_mask = torch.arange(node_num).expand(len(node_nums), node_num).cuda() < node_nums.unsqueeze(1)
        edu_attn_mask=edu_attn_mask.unsqueeze(1).expand(batch_size, node_num, node_num).reshape(batch_size*node_num, node_num)
        edu_attn_mask = StructureAwareAttention.masking_bias(edu_attn_mask)

        nodes = self.norm(dialog_input+dialog_output)
        # nodes=self.norm(dialog_output)
        nodes = nodes.unsqueeze(1).expand(batch_size, node_num, node_num, self.hidden_size)
        nodes=nodes.reshape(batch_size*node_num, node_num, self.hidden_size)
        const_path = self.path_emb(speakers, turns).unsqueeze(1).expand(batch_size, node_num, node_num, node_num, self.path_hidden_size).reshape(batch_size*node_num, node_num, node_num, self.path_hidden_size)
        struct_path=self.expand_and_mask_paths(self.path_model(graphs))
        update_mask = self.get_update_mask(batch_size, node_num)
        gnn_hx = None
        tmp=torch.arange(node_num)
        memory=[]
        for _ in range(self.layer_num):
            nodes, _ = self.gnn(nodes, edu_attn_mask, struct_path + const_path)
            gnn_hx = self.path_update(nodes, const_path, gnn_hx, update_mask)
            struct_path[update_mask] = gnn_hx
            layer_path = struct_path.reshape(batch_size, node_num, node_num, node_num, self.path_hidden_size)
            layer_path=self.get_hidden_state(layer_path)
            memory.append(layer_path)
            struct_path[update_mask]=self.dropout(struct_path[update_mask])

        struct_path = struct_path.reshape(batch_size, node_num, node_num, node_num, self.path_hidden_size)
        predicted_path = torch.cat((struct_path, struct_path.transpose(2, 3)), -1)[:, tmp, tmp]
        return self.link_classifier(predicted_path).squeeze(-1), \
               self.label_classifier(predicted_path), memory

    def get_hidden_state(self, struct_path):
        batch_size, node_num, _, _, path_hidden_size=struct_path.size()
        hidden_state=torch.zeros(batch_size, node_num, node_num, path_hidden_size).to(struct_path)
        for i in range(1, node_num):
            hidden_state[:, i, :i]=struct_path[:, i, i, :i]
            hidden_state[:, :i, i]=struct_path[:, i, :i, i]
        return hidden_state

    def get_update_mask(self, batch_size, node_num):
        paths = torch.zeros(batch_size, node_num, node_num, node_num).bool()
        for i in range(node_num):
            paths[:, i, i, :i] = True
            paths[:, i, :i, i] = True
        return paths.reshape(batch_size*node_num, node_num, node_num)

    def expand_and_mask_paths(self, paths):
        batch_size, node_num, _, path_hidden_size = paths.size()
        paths=paths.unsqueeze(1).expand(batch_size, node_num, node_num, node_num, path_hidden_size).clone()
        for i in range(node_num):
            paths[:, i, i, :i]=0
            paths[:, i, :i, i] = 0
        return paths.reshape(batch_size*node_num, node_num, node_num, path_hidden_size)


class StudentModel(nn.Module):
    def __init__(self, args, pretrained_embedding):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(embeddings=pretrained_embedding, freeze=False)
        self.sent_gru = nn.GRU(args.glove_embedding_size, args.hidden_size // 2, batch_first=True, bidirectional=True)
        self.dialog_gru = nn.GRU(args.hidden_size, args.hidden_size // 2, batch_first=True, bidirectional=True)

        self.path_emb = PathEmbedding(args)
        self.path_update = PathUpdateModel(args)

        self.gnn = StructureAwareAttention(args.hidden_size, args.path_hidden_size, args.num_heads, args.dropout)
        self.layer_num = args.num_layers  # num of layer of SSAGNN, hyperparameter T

        self.link_classifier = Classifier(args.path_hidden_size * 2, args.path_hidden_size, 1)  # MLP classifier
        self.label_classifier = Classifier(args.path_hidden_size * 2, args.path_hidden_size, args.parent_relation_dims)  # MLP classifier

        self.norm = nn.LayerNorm(args.hidden_size)
        self.dropout = nn.Dropout(args.dropout)
        self.root = nn.Parameter(torch.zeros(args.hidden_size), requires_grad=False)  # a trainable embedding for the dummy root node

        self.hidden_size = args.hidden_size
        self.path_hidden_size = args.path_hidden_size

        self.max_edu_dist = args.max_edu_dist
        self.relation_type_num = args.parent_relation_dims

    """def compute_loss(self, **kwargs):
        forward_outputs = self.forward(**kwargs)
        link_scores = forward_outputs["link_scores"]
        label_scores = forward_outputs["label_scores"]
        graphs = kwargs["graphs"]
        mask = kwargs["mask"]
        # mask = get_mask(node_num=edu_nums + 1, max_edu_dist=args.max_edu_dist).cuda()
        link_loss, label_loss, negative_loss = compute_loss(link_scores.clone(), label_scores.clone(), graphs, mask,
                                                            negative=True)
        link_loss = link_loss.mean()
        label_loss = label_loss.mean()
        loss = link_loss + label_loss + negative_loss * 0.2
        # accum_train_link_loss += link_loss.item()
        # accum_train_label_loss += label_loss.item()

        return {"link_scores": link_scores,
                "label_scores": label_scores,
                "memory": forward_outputs["memory"],
                "loss": loss,
                "link_loss": link_loss,
                "label_loss": label_loss,
                "negative_loss": negative_loss,
                }"""

    @staticmethod
    def _compute_loss(link_scores, label_scores, graphs, mask):
        # forward_outputs = self.forward(**kwargs)
        # link_scores = forward_outputs["link_scores"]
        # label_scores = forward_outputs["label_scores"]
        # graphs = kwargs["graphs"]
        # mask = kwargs["mask"]
        # mask = get_mask(node_num=edu_nums + 1, max_edu_dist=args.max_edu_dist).cuda()
        link_loss, label_loss, negative_loss = compute_loss(
            link_scores.clone(),
            label_scores.clone(),
            graphs,
            mask,
            negative=True
        )
        link_loss = link_loss.mean()
        label_loss = label_loss.mean()
        loss = link_loss + label_loss + negative_loss * 0.2
        # accum_train_link_loss += link_loss.item()
        # accum_train_label_loss += label_loss.item()
        return {
            "loss": loss,
            "link_loss": link_loss,
            "label_loss": label_loss,
            "negative_loss": negative_loss,
        }

    def predict(self, **kwargs):
        # outputs = self.forward(kwargs["texts"], kwargs["lengths"], kwargs["edu_nums"])
        outputs = self.forward(kwargs["texts"], kwargs["lengths"], kwargs["edu_nums"], kwargs["graphs"])
        # eval_link_loss, eval_label_loss = compute_loss(link_scores, label_scores, graphs, mask)
        # accum_eval_link_loss.append((eval_link_loss.sum(), eval_link_loss.size(-1)))
        # accum_eval_label_loss.append((eval_label_loss.sum(), eval_label_loss.size(-1)))

        link_scores = outputs["link_scores"]  # (batch, max_node_num, max_node_num)
        label_scores = outputs["label_scores"]  # (batch, max_node_num, max_node_num, relation_type_num)
        edu_nums = kwargs["edu_nums"]

        batch_size = link_scores.size(0)
        max_len = edu_nums.max()
        mask = get_mask(node_num=edu_nums + 1, max_edu_dist=self.max_edu_dist).cuda()
        link_scores[~mask] = -1e9
        predicted_links = torch.argmax(link_scores, dim=-1)  # (batch, max_node_num)
        predicted_labels = torch.argmax(
            label_scores.reshape(-1, max_len + 1, self.relation_type_num)
            [torch.arange(batch_size * (max_len + 1)), predicted_links.reshape(-1)].reshape(
                batch_size, max_len + 1, self.relation_type_num),
            dim=-1)  # (batch, max_node_num, relation_type_num), where each label is corresponding to predicted_links
        predicted_links = predicted_links[:, 1:] - 1  # wipe off the dummy node in each instance
        predicted_labels = predicted_labels[:, 1:]  # wipe off the dummy node in each instance

        outputs.update({
            "father_ids": predicted_links,
            "father_labels": predicted_labels,
            "previous_ids": None,
            "previous_labels": None,
        })
        return outputs


    def forward(self, texts, lengths, edu_nums, graphs=None, speakers=None, turns=None, **kwargs):
        """

        :param texts: size (batch_size, edu_num, sentence_len)
        :param lengths: size (batch_size, edu_num), dtype=torch.Long, length of each sentence
        :param edu_nums: size (batch_size), dtype=torch.Long, length of each discourse (num of sentence in each document)
        :param speakers:
        :param turns:
        :return:
        """
        batch_size, edu_num, sentence_len = texts.size()
        node_num = edu_num + 1  # add the dummy node

        texts = texts.reshape(batch_size * edu_num, sentence_len)
        texts = self.emb(texts)  # get token embeddings

        sent_output, sent_hx = self.sent_gru(texts)  # sentence-level gru encoding
        sent_output = self.dropout(sent_output)

        sent_output = sent_output.reshape(batch_size * edu_num, sentence_len, 2, -1)  # 2 for bidirectional level of gur output
        tmp = torch.arange(batch_size * edu_num)
        dialog_input = torch.cat((sent_output[tmp, lengths.reshape(-1) - 1, 0], sent_output[tmp, 0, 1]), dim=-1)
        # get the batchified sentence representation vectors
        dialog_input = torch.cat((self.root.expand(batch_size, 1, dialog_input.size(-1)),
                                  dialog_input.reshape(batch_size, edu_num, -1)),dim=1)  # (batch, max_edu_num, hidden_dim)
        # append the dummy node vector for each discourse in the batch

        dialog_output, dialog_hx = self.dialog_gru(dialog_input)  # discourse-level gru
        dialog_output = self.dropout(dialog_output)

        node_nums = edu_nums + 1  # add the dummy node
        edu_attn_mask = torch.arange(node_num).expand(len(node_nums), node_num).cuda() < node_nums.unsqueeze(1)  # attention mask for each discourse in the batch
        edu_attn_mask = StructureAwareAttention.masking_bias(edu_attn_mask)  # size (batch?, max_node_num)

        nodes = self.norm(dialog_input+dialog_output)  # nn.LayerNorm + Residual connection, (batch, max_edu_num, hidden_dim)
        if speakers != None:  # TODO
            const_path = self.path_emb(speakers, turns)  # (batch, max_node_num, max_node_num, params.path_hidden_size)
        else:
            const_path = torch.zeros(batch_size, edu_num, edu_num, self.path_hidden_size).cuda()
        struct_path = torch.zeros_like(const_path)  # (batch, max_node_num, max_node_num, params.path_hidden_size)
        memory = []
        for _ in range(self.layer_num):  # number of gnn layer, T, each layer share the same gnn parameters
            nodes, _ = self.gnn(nodes, edu_attn_mask, struct_path + const_path)
            struct_path = self.path_update(nodes, const_path, struct_path)
            memory.append(struct_path)
            struct_path = self.dropout(struct_path)
        predicted_path = torch.cat((struct_path, struct_path.transpose(1, 2)), -1)  # (batch, max_node_num, max_node_num, 2 * params.path_hidden_size)

        link_scores = self.link_classifier(predicted_path).squeeze(-1)  # (batch, max_node_num, max_node_num, 1) -> (batch, max_node_num, max_node_num)
        label_scores = self.label_classifier(predicted_path)  # (batch, max_node_num, max_node_num, relation_type_num)
        outputs = {
            "link_scores": link_scores,
            "label_scores": label_scores,
            "memory": memory,
            "loss": None,
            "link_loss": None,
            "label_loss": None,
            "negative_loss": None,
        }
        if graphs is not None:
            mask = get_mask(node_num=edu_nums + 1, max_edu_dist=self.max_edu_dist).cuda()
            outputs.update(self._compute_loss(link_scores, label_scores, graphs, mask))

        return outputs

    def get_hidden_state(self, struct_path):
        batch_size, node_num, _, _, path_hidden_size=struct_path.size()
        hidden_state=torch.zeros(batch_size, node_num, node_num, path_hidden_size).to(struct_path)
        for i in range(1, node_num):
            hidden_state[:, i, :i]=struct_path[:, i, i, :i]
            hidden_state[:, :i, i]=struct_path[:, i, :i, i]
        return hidden_state

    def get_update_mask(self, batch_size, node_num):
        paths = torch.zeros(batch_size, node_num, node_num, node_num).bool()
        for i in range(node_num):
            paths[:, i, i, :i] = True
            paths[:, i, :i, i] = True
        return paths.reshape(batch_size*node_num, node_num, node_num)

    def expand_and_mask_paths(self, paths):
        batch_size, node_num, _, path_hidden_size = paths.size()
        paths=paths.unsqueeze(1).expand(batch_size, node_num, node_num, node_num, path_hidden_size).clone()
        for i in range(node_num):
            paths[:, i, i, :i]=0
            paths[:, i, :i, i] = 0
        return paths.reshape(batch_size*node_num, node_num, node_num, path_hidden_size)


class StudentModelPLM(nn.Module):
    def __init__(self, args, config, node_encoder=None,):
        super().__init__()
        # # self.emb = nn.Embedding.from_pretrained(embeddings=pretrained_embedding, freeze=False)
        # self.sent_gru = nn.GRU(args.glove_embedding_size, args.hidden_size // 2, batch_first=True, bidirectional=True)
        # self.dialog_gru = nn.GRU(args.hidden_size, args.hidden_size // 2, batch_first=True, bidirectional=True)
        self.paragraph_encoder = node_encoder or BERTEncoder(model_name_or_path=args.model_name_or_path, config=config)
        self.encoder_hidden_dim = self.paragraph_encoder.hidden_dim
        # self.global_encoder = DocumentLevelRNNEncoder(in_dim=self.encoder_hidden_dim, hidden_dim=args.hidden_size // 2, out_dim=args.hidden_size // 2,)

        args.hidden_size = self.encoder_hidden_dim
        
        """self.path_emb = PathEmbedding(args)"""  # 1/1
        self.use_relative_position_embedding = args.use_relative_position_embedding  # 1/1
        self.path_emb_distance_only = PathEmbeddingDistance(args) if args.use_relative_position_embedding else None  # 1/1
        self.path_update = PathUpdateModel(args)

        self.gnn = StructureAwareAttention(args.hidden_size, args.path_hidden_size, args.num_heads, args.dropout)
        self.layer_num = args.num_layers  # num of layer of SSAGNN, hyperparameter T

        '''self.link_classifier = Classifier(args.path_hidden_size * 2, args.path_hidden_size, 1)  # MLP classifier, ==``concate-mlp"
        """self.label_classifier = Classifier(args.path_hidden_size * 2, args.path_hidden_size, args.parent_relation_dims)  # MLP classifier"""
        self.label_classifier = Classifier(args.path_hidden_size * 2, args.path_hidden_size, args.parent_relation_dims + 1)  # MLP classifier, 12/14'''

        self.link_classifier = ArbitraryPairPointer(
            head_input_dims=args.path_hidden_size,
            tail_input_dims=args.path_hidden_size,
            output_dims=1,
            pair_interation_method=args.pair_interation_method_parent or "concate-mlp",
            position_highlight_mlp=False,
            # relative_position_embedding=self.relative_position_embedding if args.use_relative_position_embedding else None
        )

        self.label_classifier = ArbitraryPairClassifier(
            head_input_dims=args.path_hidden_size,
            tail_input_dims=args.path_hidden_size,
            output_dims=args.parent_relation_dims + 1,
            pair_interation_method=args.pair_interation_method_parent_label or "concate-mlp",
            position_highlight_mlp=False,
            # relative_position_embedding=self.relative_position_embedding if args.use_relative_position_embedding else None
        )


        self.norm = nn.LayerNorm(args.hidden_size)
        self.dropout = nn.Dropout(args.dropout)
        self.root = nn.Parameter(torch.zeros(args.hidden_size), requires_grad=False)  # a trainable embedding for the dummy root node

        self.hidden_size = args.hidden_size
        self.path_hidden_size = args.path_hidden_size

        self.max_edu_dist = args.max_edu_dist
        self.relation_type_num = args.parent_relation_dims

        self.use_negative_loss = args.use_negative_loss
        self.negative_loss_weight = args.negative_loss_weight
        if self.use_negative_loss:
            print("use negative loss")

        #####use previous task#####
        self.use_previous_joint_loss = args.use_previous_joint_loss
        self.previous_loss_ratio = args.previous_loss_ratio
        if args.use_previous_joint_loss:
            # self.previous_relation_classifier = PairClassifier(input_dims=args.path_hidden_size * 2, combine_before=args.combine_before)
            # self.previous_relation_classifier = GraphClassifier(input_dims=args.path_hidden_size * 2, combine_before=args.combine_before)
            self.unified_previous_classifier = args.unified_previous_classifier
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

    """def compute_loss(self, **kwargs):
        forward_outputs = self.forward(**kwargs)
        link_scores = forward_outputs["link_scores"]
        label_scores = forward_outputs["label_scores"]
        graphs = kwargs["graphs"]
        mask = kwargs["mask"]
        # mask = get_mask(node_num=edu_nums + 1, max_edu_dist=args.max_edu_dist).cuda()
        link_loss, label_loss, negative_loss = compute_loss(link_scores.clone(), label_scores.clone(), graphs, mask,
                                                            negative=True)
        link_loss = link_loss.mean()
        label_loss = label_loss.mean()
        loss = link_loss + label_loss + negative_loss * 0.2
        # accum_train_link_loss += link_loss.item()
        # accum_train_label_loss += label_loss.item()

        return {"link_scores": link_scores,
                "label_scores": label_scores,
                "memory": forward_outputs["memory"],
                "loss": loss,
                "link_loss": link_loss,
                "label_loss": label_loss,
                "negative_loss": negative_loss,
                }"""

    @staticmethod
    def _compute_loss(link_scores, label_scores, graphs, mask, negative_loss=True, negative_weight=0.2):
        #$$# for debug check
        #$$# print("[*] _compute_loss(), called by forward(), if graph is not None")
        """
        :param link_scores: (batch, max_node_num, max_node_num)
        :param label_scores: (batch, max_node_num, max_node_num, relation_type_num)
        :param graphs: (batch, num_node, num_node) 
        :param mask: (batch, max_node_num, max_node_num)
        """
        # forward_outputs = self.forward(**kwargs)
        # link_scores = forward_outputs["link_scores"]
        # label_scores = forward_outputs["label_scores"]
        # graphs = kwargs["graphs"]
        # mask = kwargs["mask"]
        # mask = get_mask(node_num=edu_nums + 1, max_edu_dist=args.max_edu_dist).cuda()
        # print(link_scores)
        # print(label_scores)
        # print(graphs)
        link_loss, label_loss, negative_loss = compute_loss(
            link_scores.clone(),
            label_scores.clone(),
            graphs,
            mask,
            negative=negative_loss,
        )
        
        link_loss = link_loss.mean()
        label_loss = label_loss.mean()
        # print("label_loss", label_loss)
        
        loss = link_loss + label_loss
        # accum_train_link_loss += link_loss.item()
        # accum_train_label_loss += label_loss.item()
        outputs = {
            "loss": loss,
            "link_loss": link_loss,
            "label_loss": label_loss,
            
        }
        if negative_loss is not None:
            loss += negative_loss * negative_weight
            outputs.update(
                {"negative_loss": negative_loss,}
            )
        return outputs

    def predict(self, **kwargs):
        # outputs = self.forward(kwargs["texts"], kwargs["lengths"], kwargs["edu_nums"])
        # outputs = self.forward(kwargs["input_ids"], kwargs["lengths"], kwargs["edu_nums"], kwargs["input_mask"],)# kwargs["graphs"])
        kwargs["graphs"]=None
        kwargs["golden_previous_ids"]=None
        kwargs["golden_previous_labels"]=None
        outputs = self.forward(**kwargs)  # 11/21

        # eval_link_loss, eval_label_loss = compute_loss(link_scores, label_scores, graphs, mask)
        # accum_eval_link_loss.append((eval_link_loss.sum(), eval_link_loss.size(-1)))
        # accum_eval_label_loss.append((eval_label_loss.sum(), eval_label_loss.size(-1)))
        ##### edu_nums = kwargs["edu_nums"]

        """link_scores = outputs["link_scores"]  # (batch, max_node_num, max_node_num)
        label_scores = outputs["label_scores"]  # (batch, max_node_num, max_node_num, relation_type_num)
        edu_nums = kwargs["edu_nums"]
        batch_size = link_scores.size(0)
        max_len = edu_nums.max()
        
        mask = get_mask(node_num=edu_nums + 1, max_edu_dist=self.max_edu_dist).cuda()
        link_scores[~mask] = -1e9
        predicted_links = torch.argmax(link_scores, dim=-1)  # (batch, max_node_num)
        
        predicted_labels = torch.argmax(
            label_scores.reshape(-1, max_len + 1, self.relation_type_num)
            [torch.arange(batch_size * (max_len + 1)), predicted_links.reshape(-1)].reshape(
                batch_size, max_len + 1, self.relation_type_num),
            dim=-1)  # (batch, max_node_num, relation_type_num), where each label is corresponding to predicted_links
        predicted_links = predicted_links[:, 1:] - 1  # wipe off the dummy node in each instance
        predicted_labels = predicted_labels[:, 1:]  # wipe off the dummy node in each instance
        predicted_labels[predicted_labels==0] = self.relation_type_num  # TODO 1107
        predicted_labels = predicted_labels - 1 # TODO 1107
        predicted_links[predicted_labels==(self.relation_type_num-1)] = -1   # TODO 1107

        outputs.update({
            "father_ids": predicted_links,
            "father_labels": predicted_labels,
            "previous_ids": None,
            "previous_labels": None,
        })
        # print(outputs)
        return outputs"""
        ##### return self._get_predicted_values(outputs, edu_nums)
        return outputs
    
    def _get_predicted_values(self, outputs, edu_nums):
        #$$# for debug check
        #$$# print("[*] _get_predicted_values(), called by forward(), always called")
        link_scores = outputs["link_scores"]  # (batch, max_node_num, max_node_num)
        label_scores = outputs["label_scores"]  # (batch, max_node_num, max_node_num, relation_type_num)
        # edu_nums = kwargs["edu_nums"]
        batch_size = link_scores.size(0)
        max_len = edu_nums.max()
        
        # mask = get_mask(node_num=edu_nums + 1, max_edu_dist=self.max_edu_dist).cuda()
        mask = get_mask(node_num=edu_nums + 1, max_edu_dist=self.max_edu_dist).to(link_scores.device)
        #$$# print(f"mask: {mask}, {mask.shape}")

        link_scores[~mask] = -1e9
        #$$# print(f"link_scores[~mask] = -1e9: {link_scores}, {link_scores.shape}")
        predicted_links = torch.argmax(link_scores, dim=-1)  # (batch, max_node_num)
        #$$# print(f"predicted_links = torch.argmax(link_scores, dim=-1): {predicted_links}, {predicted_links.shape}")
        
        """predicted_labels = torch.argmax(
            label_scores.reshape(-1, max_len + 1, self.relation_type_num)
            [torch.arange(batch_size * (max_len + 1)), predicted_links.reshape(-1)].reshape(
                batch_size, max_len + 1, self.relation_type_num),
            dim=-1)  # (batch, max_node_num, relation_type_num), where each label is corresponding to predicted_links"""
        
        #$$# print(f"label_scores: {label_scores}, {label_scores.shape}")
        #$$# print(f"label_scores.reshape(-1, max_len + 1, self.relation_type_num+1): {label_scores.reshape(-1, max_len + 1, self.relation_type_num+1)}, {label_scores.reshape(-1, max_len + 1, self.relation_type_num+1).shape}")
        #$$# print(f"mask ids of predicted labels [torch.arange(batch_size * (max_len + 1)), predicted_links.reshape(-1)]: {torch.arange(batch_size * (max_len + 1))}, {torch.arange(batch_size * (max_len + 1)).shape}")
        #$$# print(f"mask ids of predicted labels [torch.arange(batch_size * (max_len + 1)), predicted_links.reshape(-1)]: {predicted_links.reshape(-1)}, {predicted_links.reshape(-1).shape}")

        if self.use_negative_loss:
            #$$# print("use_negative_loss, so predicted labels start at id 0, where 0 is the 'NA' label")
            predicted_labels = torch.argmax(
                label_scores.reshape(-1, max_len + 1, self.relation_type_num+1)
                [torch.arange(batch_size * (max_len + 1)), predicted_links.reshape(-1)].reshape(
                    batch_size, max_len + 1, self.relation_type_num+1),
                dim=-1)  # (batch, max_node_num, relation_type_num), where each label is corresponding to predicted_links
            #$$# print(f"predicted_label, before -1: {predicted_labels}, {predicted_labels.shape}")
        else:
            #$$# print("not use_negative_loss, so predicted labels start at id 1, where 0 is the 'NA' label")
            predicted_labels = torch.argmax(
                label_scores.reshape(-1, max_len + 1, self.relation_type_num+1)
                [torch.arange(batch_size * (max_len + 1)), predicted_links.reshape(-1)].reshape(
                    batch_size, max_len + 1, self.relation_type_num+1)[:, :, 1:],
                dim=-1) + 1  # +1, 12/15
            #$$# print(f"predicted_label, before -1: {predicted_labels}, {predicted_labels.shape}")
        
        # predicted_previous = torch.tensor(father_id_to_previous_id(predicted_links[:, 1:].detach().tolist())[:, 1:]) - 1

        predicted_previous = father_id_to_previous_id(predicted_links[:, 1:].detach().tolist())
        # predicted_previous = predicted_previous[:, 1:]
        predicted_previous = torch.tensor(predicted_previous)[:, 1:] - 1
        
        predicted_links = predicted_links[:, 1:] - 1  # wipe off the dummy node in each instance
        #$$# print("predicted_links wipe off the dummy node and -1", predicted_links, predicted_links.shape)
        predicted_labels = predicted_labels[:, 1:]  # wipe off the dummy node in each instance
        """predicted_labels[predicted_labels==0] = self.relation_type_num"""  # TODO 1107
        predicted_labels = predicted_labels - 1 # TODO 1107
        #$$# print("predicted_labels wipe off the dummy node and -1", predicted_labels, predicted_labels.shape)
        """predicted_links[predicted_labels==(self.relation_type_num-1)] = -1"""   # TODO 1107
        if self.use_negative_loss:
            #$$# print("use_negative_loss, so predicted labels 'NA' (-1) would change the predicted links also be -1")
            predicted_links[predicted_labels==-1] = -1 
            #$$# print(f"predicted_links[predicted_labels==-1] = -1: {predicted_links}, {predicted_links.shape}")
        predicted_labels[predicted_links==-1] = -1

        outputs.update({
            "father_ids": predicted_links,
            "father_labels": predicted_labels,
            "previous_ids": predicted_previous,
            "previous_labels": None,
        })
        # print(outputs)
        return outputs


    def forward(self, input_ids, lengths, edu_nums, input_mask=None, graphs=None, speakers=None, turns=None, golden_previous_ids=None, golden_previous_labels=None, **kwargs):
        """

        :param texts: size (batch_size, edu_num, sentence_len)
        :param lengths: size (batch_size, edu_num), dtype=torch.Long, length of each sentence
        :param edu_nums: size (batch_size), dtype=torch.Long, length of each discourse (num of sentence in each document)
        :param speakers:
        :param turns:
        :return:
        """
        # print("before encoder:", torch.cuda.memory_allocated(input_ids.device))
        """batch_size, edu_num, sentence_len = input_ids.size()"""
        batch_size = input_ids.size()[0]  # 12/26
        # print(edu_nums)
        """node_num = edu_num + 1  # add the dummy node"""
        node_nums = edu_nums + 1  # add the dummy node
        edu_num = edu_nums.max().tolist()  # 12/28
        node_num = node_nums.max().tolist()  # 12/26

        # texts = input_ids.reshape(batch_size * edu_num, sentence_len)
        # texts = self.emb(texts)  # get token embeddings
        texts = self.paragraph_encoder(input_ids, input_mask, **kwargs) 
        # print("after encoder:", torch.cuda.memory_allocated(input_ids.device))

        """sent_output, sent_hx = self.sent_gru(texts)  # sentence-level gru encoding
        sent_output = self.dropout(sent_output)

        sent_output = sent_output.reshape(batch_size * edu_num, sentence_len, 2, -1)  # 2 for bidirectional level of gur output
        tmp = torch.arange(batch_size * edu_num)
        dialog_input = torch.cat((sent_output[tmp, lengths.reshape(-1) - 1, 0], sent_output[tmp, 0, 1]), dim=-1)
        # get the batchified sentence representation vectors
        dialog_input = torch.cat((self.root.expand(batch_size, 1, dialog_input.size(-1)),
                                  dialog_input.reshape(batch_size, edu_num, -1)),dim=1)  # (batch, max_edu_num, hidden_dim)
        # append the dummy node vector for each discourse in the batch

        dialog_output, dialog_hx = self.dialog_gru(dialog_input)  # discourse-level gru"""
        texts = torch.cat((self.root.expand(batch_size, 1, texts.size(-1)),
                                  texts.reshape(batch_size, edu_num, -1)),dim=1)  # (batch, max_edu_num+1, hidden_dim)


        # dialog_output = self.global_encoder(texts, edu_nums)
        # dialog_output = self.global_encoder(texts, node_nums)
        # dialog_output = self.dropout(dialog_output)  # (Batch, max_node_num, hidden_dim)
        dialog_output = self.dropout(texts)  # (Batch, max_node_num, hidden_dim)

        
        edu_attn_mask = torch.arange(node_num).expand(len(node_nums), node_num).to(input_ids.device) < node_nums.unsqueeze(1)  # attention mask for each discourse in the batch
        """
        >>> torch.arange(6).expand(4, 6) < torch.tensor([[3],[2],[4],[1]])
        tensor([[ True,  True,  True, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True,  True,  True, False, False],
                [ True, False, False, False, False, False]])
        """
        edu_attn_mask = StructureAwareAttention.masking_bias(edu_attn_mask)  # size (batch, max_node_num+1) ->  (batch, 1, 1, max_node_num+1)

        # nodes = self.norm(dialog_input+dialog_output)  # nn.LayerNorm + Residual connection, (batch, max_edu_num, hidden_dim)
        nodes = self.norm(texts + dialog_output)  # nn.LayerNorm + Residual connection, (batch, max_edu_num, hidden_dim)
        
        if speakers != None:  # TODO
            const_path = self.path_emb(speakers, turns)  # (batch, max_node_num, max_node_num, params.path_hidden_size)
        else:
            # const_path = torch.zeros(batch_size, node_num, node_num, self.path_hidden_size).cuda()
            if self.use_relative_position_embedding:  # 1/1
                const_path = self.path_emb_distance_only(batch_size, node_num,)
            else:
                const_path = torch.zeros(batch_size, node_num, node_num, self.path_hidden_size).to(input_ids.device)

        struct_path = torch.zeros_like(const_path)  # (batch, max_node_num, max_node_num, params.path_hidden_size)
        
        memory = []
        for _ in range(self.layer_num):  # number of gnn layer, T, each layer share the same gnn parameters
            nodes, _ = self.gnn(nodes, edu_attn_mask, struct_path + const_path)
            struct_path = self.path_update(nodes, const_path, struct_path)
            memory.append(struct_path)
            struct_path = self.dropout(struct_path)
        predicted_path = torch.cat((struct_path, struct_path.transpose(1, 2)), -1)  # (batch, max_node_num, max_node_num, 2 * params.path_hidden_size)

        """link_scores = self.link_classifier(predicted_path).squeeze(-1)  # (batch, max_node_num, max_node_num, 1) -> (batch, max_node_num, max_node_num)
        label_scores = self.label_classifier(predicted_path)  # (batch, max_node_num, max_node_num, relation_type_num)"""
        link_scores = self.link_classifier.forward_with_graph_encodings_return_logits(struct_path)  # (batch, max_node_num, max_node_num)
        label_scores = self.label_classifier.forward_with_graph_encodings_return_logits(struct_path)


        outputs = {
            "link_scores": link_scores,
            "label_scores": label_scores,
            "memory": memory,
            # "loss": None,
            # "link_loss": None,
            # "label_loss": None,
            # "negative_loss": None,
        }
        if graphs is not None:
            # print("compute loss")
            # mask = get_mask(node_num=edu_nums+1, max_edu_dist=self.max_edu_dist).cuda()
            mask = get_mask(node_num=edu_nums+1, max_edu_dist=self.max_edu_dist).to(input_ids.device)
            outputs.update(self._compute_loss(link_scores, label_scores, graphs, mask, self.use_negative_loss, self.negative_loss_weight))

        
        outputs = self._get_predicted_values(outputs, edu_nums)
        #####use previous task#####
        if self.use_previous_joint_loss:
            # outputs = self._get_predicted_values(outputs, edu_nums)
            if self.unified_previous_classifier:  # 1/1
                outputs = predict_previous_relations_forward_with_graph_encodings(
                    outputs,
                    classifier=self.previous_relation_classifier,
                    encodings=struct_path,  # (batch, max_node_num, max_node_num, 2 * params.path_hidden_size)
                    predicted_father_ids=outputs["father_ids"] if "father_ids" in outputs.keys() else None,
                    golden_previous_ids=golden_previous_ids,
                    golden_previous_labels=golden_previous_labels,  # TODO
                    loss_ratio=self.previous_loss_ratio,
                )
            else:
                outputs = predict_previous_relations(
                    outputs,
                    classifier=self.previous_relation_classifier,
                    encodings=predicted_path,  # (batch, max_node_num, max_node_num, 2 * params.path_hidden_size)
                    predicted_father_ids=outputs["father_ids"] if "father_ids" in outputs.keys() else None,
                    golden_previous_ids=golden_previous_ids,
                    golden_previous_labels=golden_previous_labels,  # TODO
                    loss_ratio=self.previous_loss_ratio,
                )
        #####use previous task#####
        
        # print("before reture", torch.cuda.memory_allocated(input_ids.device))
        return outputs

    def get_hidden_state(self, struct_path):
        batch_size, node_num, _, _, path_hidden_size=struct_path.size()
        hidden_state=torch.zeros(batch_size, node_num, node_num, path_hidden_size).to(struct_path)
        for i in range(1, node_num):
            hidden_state[:, i, :i]=struct_path[:, i, i, :i]
            hidden_state[:, :i, i]=struct_path[:, i, :i, i]
        return hidden_state

    def get_update_mask(self, batch_size, node_num):
        paths = torch.zeros(batch_size, node_num, node_num, node_num).bool()
        for i in range(node_num):
            paths[:, i, i, :i] = True
            paths[:, i, :i, i] = True
        return paths.reshape(batch_size*node_num, node_num, node_num)

    def expand_and_mask_paths(self, paths):
        batch_size, node_num, _, path_hidden_size = paths.size()
        paths=paths.unsqueeze(1).expand(batch_size, node_num, node_num, node_num, path_hidden_size).clone()
        for i in range(node_num):
            paths[:, i, i, :i]=0
            paths[:, i, :i, i] = 0
        return paths.reshape(batch_size*node_num, node_num, node_num, path_hidden_size)

class Bridge(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bridge=nn.Linear(args.path_hidden_size, args.path_hidden_size)

    def forward(self, input):
        return self.bridge(input)


class StructureAwareAttention(nn.Module):
    def __init__(self, hidden_size, path_hidden_size, head_num, dropout):
        super(StructureAwareAttention, self).__init__()
        self.q_transform = nn.Linear(hidden_size, hidden_size)
        self.k_transform = nn.Linear(hidden_size, hidden_size)
        self.v_transform = nn.Linear(hidden_size, hidden_size)
        self.struct_k_transform = nn.Linear(path_hidden_size, hidden_size // head_num)
        self.struct_v_transform = nn.Linear(path_hidden_size, hidden_size // head_num)
        self.o_transform = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()
        self.hidden_size = hidden_size
        self.head_num = head_num
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.path_norm = nn.LayerNorm(path_hidden_size)

    def forward(self, nodes, bias, paths):
        """
        :param nodes: (batch, max_node_num, hidden_dim)
        :param bias: (batch, 1, 1, max_node_num+1)
        :param paths: (batch, max_node_num, max_node_num, args.path_hidden_size)
        :return:
        """
        q, k, v = self.q_transform(nodes), self.k_transform(nodes), self.v_transform(nodes)
        q = self.split_heads(q, self.head_num)  # this operation is of the same effect as Multiple-Head-Attention (batch, max_node_num, hidden_dim) -> (batch, num_heads, max_node_num, channel_dim)
        k = self.split_heads(k, self.head_num)
        v = self.split_heads(v, self.head_num)
        
        paths = self.path_norm(paths)
        struct_k, struct_v = self.struct_k_transform(paths), self.struct_v_transform(paths)
        
        q = q * (self.hidden_size // self.head_num) ** -0.5
        w = torch.matmul(q, k.transpose(-1, -2))
        struct_w = torch.matmul(q.transpose(1,2), struct_k.transpose(-1, -2)).transpose(1,2)
        # print(w.shape)  # (batch_size, head_num, max_node_num, max_node_num)
        # print(struct_w.shape)  # (batch_size, head_num, max_node_num, max_node_num)
        # print(bias.shape)  # (batch_size, 1, 1, max_node_num)
        w = w + struct_w + bias
        w = torch.nn.functional.softmax(w, dim=-1)
        output = torch.matmul(w, v)+torch.matmul(w.transpose(1,2), struct_v).transpose(1,2)
        output = self.activation(self.o_transform(self.combine_heads(output)))
        return self.norm(nodes + self.dropout(output)), w

    @staticmethod
    def split_heads(x, heads):
        batch = x.shape[0]
        length = x.shape[1]
        channels = x.shape[2]

        y = torch.reshape(x, [batch, length, heads, channels // heads])
        return torch.transpose(y, 2, 1)

    @staticmethod
    def combine_heads(x):
        batch = x.shape[0]
        heads = x.shape[1]
        length = x.shape[2]
        channels = x.shape[3]

        y = torch.transpose(x, 2, 1)

        return torch.reshape(y, [batch, length, heads * channels])

    @staticmethod
    def masking_bias(mask, inf=-1e9):
        ret = ~mask * inf  # True -> False; False -> True
        return torch.unsqueeze(torch.unsqueeze(ret, 1),1)


class PathUpdateModel(nn.Module):
    def __init__(self, args):
        super(PathUpdateModel, self).__init__()
        self.x_dim = args.hidden_size
        self.h_dim = args.path_hidden_size

        self.r = nn.Linear(2*self.x_dim + self.h_dim, self.h_dim, True)
        self.z = nn.Linear(2*self.x_dim + self.h_dim, self.h_dim, True)

        self.c = nn.Linear(2*self.x_dim, self.h_dim, True)
        self.u = nn.Linear(self.h_dim, self.h_dim, True)

    def forward(self, nodes, bias, hx, mask=None):
        """

        :param nodes: nodes
        :param bias: constant_path
        :param hx: structure_path
        :param mask:
        :return:
        """
        batch_size, node_num, hidden_size = nodes.size()
        nodes = nodes.unsqueeze(1).expand(batch_size, node_num, node_num, hidden_size)
        nodes = torch.cat((nodes, nodes.transpose(1, 2)),dim=-1)
        if mask is not None:
            nodes, bias =nodes[mask], bias[mask]
        if hx is None:
            hx=torch.zeros_like(bias)

        rz_input = torch.cat((nodes, hx), -1)
        r = torch.sigmoid(self.r(rz_input))
        z = torch.sigmoid(self.z(rz_input))

        u = torch.tanh(self.c(nodes) + r * self.u(hx))

        new_h = z * hx + (1 - z) * u
        return new_h


class PathEmbedding(nn.Module):
    def __init__(self, args):
        super(PathEmbedding, self).__init__()
        self.speaker = nn.Embedding(2, args.path_hidden_size // 4)
        self.turn = nn.Embedding(2, args.path_hidden_size // 4)
        self.valid_dist = args.valid_dist  # int, default=4
        self.position = nn.Embedding(self.valid_dist * 2 + 3, args.path_hidden_size // 2)
        # self.valid_dist * 2 + 3 is equal to len(range(-self.valid_dist - 1, self.valid_dist + 1+1))

        self.tmp = torch.arange(200)
        self.path_pool = self.tmp.expand(200, 200) - self.tmp.unsqueeze(1)
        self.path_pool[self.path_pool > self.valid_dist] = self.valid_dist + 1
        self.path_pool[self.path_pool < -self.valid_dist] = -self.valid_dist - 1
        self.path_pool += self.valid_dist + 1

    def forward(self, speaker, turn):
        """

        :param speaker: (batch, max_node_num, max_node_num), 0/1 tensor, indicating the same-speaker nodes
        :param turn: (batch, max_node_num, max_node_num), 0/1 tensor, indicating the same-dialogue-turn nodes
        :return: initial representation of node-node pairs, concationation of 3 kinds of embeddings,
        i.e. same-speaker embedding, same-turn embedding and relative-positional embedding
        size (batch, max_node_num, max_node_num, params.path_hidden_size)
        """
        batch_size, node_num, _ = speaker.size()
        speaker = self.speaker(speaker)  # (batch, max_node_num, max_node_num, params.path_hidden_size // 4)
        turn = self.turn(turn)  # (batch, max_node_num, max_node_num, params.path_hidden_size // 4)
        # position = self.position(self.path_pool[:node_num, :node_num].cuda())  # (max_node_num, max_node_num, params.path_hidden_size // 2)
        position = self.position(self.path_pool[:node_num, :node_num])  # (max_node_num, max_node_num, params.path_hidden_size // 2)
        position = position.expand(batch_size, node_num, node_num, position.size(-1))
        # (batch, max_node_num, max_node_num, params.path_hidden_size // 2),
        # relative position embedding for each node, relative to other nodes in the same dialogue
        return torch.cat((speaker, turn, position), dim=-1)


class PathEmbeddingDistance(nn.Module):
    # 1/1
    def __init__(self, args):
        super(PathEmbeddingDistance, self).__init__()
        """self.speaker = nn.Embedding(2, args.path_hidden_size // 4)
        self.turn = nn.Embedding(2, args.path_hidden_size // 4)"""
        self.valid_dist = args.valid_dist  # int, default=4
        """self.position = nn.Embedding(self.valid_dist * 2 + 3, args.path_hidden_size // 2)"""
        self.position = nn.Embedding(self.valid_dist * 2 + 3, args.path_hidden_size)
        # self.valid_dist * 2 + 3 is equal to len(range(-self.valid_dist - 1, self.valid_dist + 1+1))

        self.tmp = torch.arange(args.max_paragraph_num)
        self.path_pool = self.tmp.expand(args.max_paragraph_num, args.max_paragraph_num) - self.tmp.unsqueeze(1)
        """
        >>> torch.arange(6).expand(6,6) - torch.arange(6).unsqueeze(1)
        tensor([[ 0,  1,  2,  3,  4,  5],
                [-1,  0,  1,  2,  3,  4],
                [-2, -1,  0,  1,  2,  3],
                [-3, -2, -1,  0,  1,  2],
                [-4, -3, -2, -1,  0,  1],
                [-5, -4, -3, -2, -1,  0]])
        This is to get relative distance
        """
        self.path_pool[self.path_pool > self.valid_dist] = self.valid_dist + 1
        self.path_pool[self.path_pool < -self.valid_dist] = -self.valid_dist - 1
        self.path_pool += self.valid_dist + 1
        self.path_pool = self.path_pool.to(args.device_id)  # 1/1 # TODO

    def forward(self, batch_size, node_num, speaker=None, turn=None):
        """
        :param speaker: (batch, max_node_num, max_node_num), 0/1 tensor, indicating the same-speaker nodes
        :param turn: (batch, max_node_num, max_node_num), 0/1 tensor, indicating the same-dialogue-turn nodes
        :return: initial representation of node-node pairs, concationation of 3 kinds of embeddings,
        i.e. same-speaker embedding, same-turn embedding and relative-positional embedding
        size (batch, max_node_num, max_node_num, params.path_hidden_size)
        """

        """batch_size, node_num, _ = speaker.size()
        speaker = self.speaker(speaker)  # (batch, max_node_num, max_node_num, params.path_hidden_size // 4)
        turn = self.turn(turn)  # (batch, max_node_num, max_node_num, params.path_hidden_size // 4)"""
        # position = self.position(self.path_pool[:node_num, :node_num].cuda())  # (max_node_num, max_node_num, params.path_hidden_size // 2)
        position = self.position(self.path_pool[:node_num, :node_num])  # (max_node_num, max_node_num, params.path_hidden_size // 2)
        position = position.expand(batch_size, node_num, node_num, position.size(-1))
        # (batch, max_node_num, max_node_num, params.path_hidden_size // 2),
        # relative position embedding for each node, relative to other nodes in the same dialogue
        """return torch.cat((speaker, turn, position), dim=-1)"""
        return position


class PathModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.path_hidden_size = args.path_hidden_size
        self.type_num = args.parent_relation_dims
        self.spec_type = nn.Parameter(torch.zeros(1, args.path_hidden_size), requires_grad=False)
        self.normal_type = nn.Parameter(torch.empty(args.parent_relation_dims - 1, args.path_hidden_size),
                                        requires_grad=True)
        self.dropout = nn.Dropout(0.1)
        self.reset_parameters()

    def forward(self, graphs):
        label_embedding = torch.cat((self.spec_type, self.normal_type), dim=0)
        graphs=graphs+graphs.transpose(1,2)
        path = self.dropout(nn.functional.embedding(graphs, weight=label_embedding, padding_idx=0))
        return path

    def reset_parameters(self):
        nn.init.normal_(self.normal_type, mean=0.0,
                        std=self.path_hidden_size ** -0.5)


class PathClassifier(nn.Module):
    def __init__(self, args):
        super(PathClassifier, self).__init__()
        self.type_num = args.parent_relation_dims
        self.classifier = Classifier(args.path_hidden_size, args.path_hidden_size, args.parent_relation_dims)

    def forward(self, path, target, mask):
        path = self.classifier(path)[mask]
        target = target[mask]
        # weight = torch.ones(self.type_num).float().cuda()
        weight = torch.ones(self.type_num).float().to(target.device)
        weight[0] /= target.size(0)**0.5
        return torch.nn.functional.cross_entropy(path, target, weight=weight, reduction='mean')


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super().__init__()
        self.input_transform = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh())
        self.output_transform = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        return self.output_transform(self.input_transform(x))


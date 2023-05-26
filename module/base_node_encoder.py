import torch
import torch.nn as nn

from module import prepare_html_tag_embedding, prepare_xpath_encoder, prepare_plm_text_encoder, prepare_position_embedding
# from module import prepare_aggregation_method  # 12/26
from module import DocumentLevelRNNEncoder, DocumentLevelTransformerEncoder

class BaseNodeEncoder(nn.Module):
    def __init__(self, args, config, data_processor):
        super().__init__()

        self.no_text_information = args.no_text_information
        self.paragraph_encoder = prepare_plm_text_encoder(args, config) if not self.no_text_information else None
        self.hidden_dim = self.paragraph_encoder.hidden_dim if not self.no_text_information else 0
        if self.no_text_information:  # 11/20
            assert args.use_xpath_embedding
        
        #####use html tag and XPath features#####
        # self._add_html_tag_embedding(args, html_vocab)
        # self._add_xpath_embedding(args, )
        self.use_html_embedding = args.use_html_embedding #and (html_tag_embedding is not None)
        self.use_xpath_embedding = args.use_xpath_embedding #and (xpath_encoder is not None) 
        self.use_position_embedding = args.use_position_embedding
        self.use_global_encoder = args.use_global_encoder


        self.html_tag_embedding = prepare_html_tag_embedding(args, data_processor.get_html_vocab()) if self.use_html_embedding else None
        self.html_tag_embedding_dim = self.html_tag_embedding.embedding_dim if self.use_html_embedding else 0
        # self.html_tag_embedding = html_tag_embedding

        self.xpath_encoder = prepare_xpath_encoder(args, data_processor.get_xpath_vocab()) if self.use_xpath_embedding else None
        self.xpath_embedding_dim = self.xpath_encoder.embedding_dim if self.use_xpath_embedding else 0
        # self.xpath_encoder = xpath_encoder
        
        # if self.use_html_embedding:
        #     self.hidden_dim += self.html_tag_embedding.embedding_dim
        # if self.use_xpath_embedding:
        #     self.hidden_dim += self.xpath_encoder.embedding_dim
        #####use html tag and XPath features#####

        self.relative_posotion_embedding = prepare_position_embedding(args) if self.use_position_embedding else None  # TODO 目前无效
        self.posotion_embedding_dim = self.relative_posotion_embedding.embedding_dim if self.use_position_embedding else 0
        # self.use_relative_position = args.use_relative_position
        # self.relative_position_encoding_dim = args.relative_position_encoding_dim if self.use_relative_position else 0
        # self.relative_posotion_embedding = nn.Embedding(args.max_paragraph_num, args.relative_position_encoding_dim) if self.use_global_encoder else None
    

        # self.use_global_encoder = args.use_global_encoder
        """self.global_encoder = DocumentLevelRNNEncoder(model_type=args.global_encoder_type, in_dim=self.paragraph_encoder.hidden_dim, hidden_dim=self.paragraph_encoder.hidden_dim//2, num_layers=1, bidirectional=True) if args.use_global_encoder else None  # 1108
        self.context_integrator = context_integrator or LSTMIntegrator(self.paragraph_encoder.hidden_dim, args.hidden_dim)"""
        
        
        # self.aggregation_method = "concat-project"  # ["concat", "add"]
        self.aggregation_method = args.modality_aggregation_method  # 12/26  # ["concat", "concat-project", "add", "linear-combination-single-weight", "linear-combination-complex-weight"]
        self.residual_connect = True

        """if self.aggregation_method == "concat":
            self.final_hidden_dim = self.hidden_dim + self.html_tag_embedding_dim + self.xpath_embedding_dim + self.posotion_embedding_dim
            self.projector = lambda x:x
        elif self.aggregation_method == "concat-project":
            self.final_hidden_dim = 768  # TODO
            self.projector = nn.Linear(self.hidden_dim + self.html_tag_embedding_dim + self.xpath_embedding_dim + self.posotion_embedding_dim, self.final_hidden_dim)"""
        self.final_hidden_dim, self.do_aggregation = self.prepare_aggregation_method()


        self.hidden_dim = self.final_hidden_dim
        """self.global_encoder = DocumentLevelRNNEncoder(model_type=args.global_encoder_type, in_dim=self.hidden_dim, hidden_dim=self.hidden_dim//2, num_layers=1, bidirectional=True) if args.use_global_encoder else None  # 1108"""
        if args.use_global_encoder:  # 12/26
            if args.global_encoder_type in ["rnn","lstm","gru"]:
                self.global_encoder = DocumentLevelRNNEncoder(model_type=args.global_encoder_type, in_dim=self.hidden_dim, hidden_dim=self.hidden_dim//2, num_layers=1, bidirectional=True)
            else:
                self.global_encoder = DocumentLevelTransformerEncoder(in_dim=self.hidden_dim, out_dim=self.hidden_dim, num_layers=3,)


        print(f"BaseNodeEncoder: paragraph_encoder={not self.no_text_information}; html_tag_emb={self.use_html_embedding}; xpath_emb={self.use_xpath_embedding}; position_emb={self.use_position_embedding}; global_enc={self.use_global_encoder} | final_hidden_dim={self.hidden_dim}")
        

    # @staticmethod
    # def prepare_text_encoder(args):
    #     text_encoder = BERTEncoder(config=config)
    #     return text_encoder

    # @staticmethod
    # def prepare_position_embedding(args):
    #     pass

    # @staticmethod
    # def prepare_htag_embedding(args):
    #     pass
    
    # @staticmethod
    # def prepare_xpath_encoder(args):
    #     pass

    def do_aggregation_concate(self, para_embeddings, html_embeddings, xpath_embeddings, position_embeddings):
        para_embeddings = torch.cat((para_embeddings, html_embeddings, xpath_embeddings, position_embeddings), dim=-1)
        para_embeddings = self.projector(para_embeddings)
        return para_embeddings
    
    def do_aggregation_concat_project(self, para_embeddings, html_embeddings, xpath_embeddings, position_embeddings):
        para_embeddings = torch.cat((para_embeddings, html_embeddings, xpath_embeddings, position_embeddings), dim=-1)
        para_embeddings = self.projector(para_embeddings)
        return para_embeddings
    
    def do_aggregation_add(self, para_embeddings, html_embeddings, xpath_embeddings, position_embeddings):
        para_embeddings = self.hidden_projector(para_embeddings)
        html_embeddings = self.html_tag_embedding_projector(html_embeddings)
        xpath_embeddings = self.xpath_embedding_projector(xpath_embeddings)
        position_embeddings = self.posotion_embedding_projector(position_embeddings)
        if self.no_text_information:  # 11/20
            #assert args.use_xpath_embedding
            base_embeddings = xpath_embeddings
        else:
            base_embeddings = para_embeddings
            if self.xpath_embedding_dim > 0:
                base_embeddings = base_embeddings + xpath_embeddings
        if self.html_tag_embedding_dim > 0:
            base_embeddings = base_embeddings + html_embeddings
        if self.posotion_embedding_dim > 0:
            base_embeddings = base_embeddings + position_embeddings

        return base_embeddings
    
    def do_aggregation_lc_complex(self, para_embeddings, html_embeddings, xpath_embeddings, position_embeddings):
        weights = self.weight_linear(torch.cat((para_embeddings, html_embeddings, xpath_embeddings, position_embeddings), dim=-1))
        if self.no_text_information:
            weights[0] = -1e8
        if self.xpath_embedding_dim == 0:
            weights[1] = -1e8
        if self.html_tag_embedding_dim == 0:
            weights[2] = -1e8
        if self.posotion_embedding_dim == 0:
            weights[3] = -1e8
        weights = self.aggregation_softmax(weights)

        para_embeddings = self.hidden_projector(para_embeddings)
        html_embeddings = self.html_tag_embedding_projector(html_embeddings)
        xpath_embeddings = self.xpath_embedding_projector(xpath_embeddings)
        position_embeddings = self.posotion_embedding_projector(position_embeddings)
        if self.no_text_information:  # 11/20
            #assert args.use_xpath_embedding
            base_embeddings = xpath_embeddings * weights[1]
        else:
            base_embeddings = para_embeddings * weights[0]
            if self.xpath_embedding_dim > 0:
                base_embeddings = base_embeddings + xpath_embeddings * weights[1]
        if self.html_tag_embedding_dim > 0:
            base_embeddings = base_embeddings + html_embeddings * weights[2]
        if self.posotion_embedding_dim > 0:
            base_embeddings = base_embeddings + position_embeddings * weights[3]

        return base_embeddings

    def do_aggregation_lc_simple(self, para_embeddings, html_embeddings, xpath_embeddings, position_embeddings):
        weights = self.aggregation_softmax(self.aggregation_weights)
        para_embeddings = self.hidden_projector(para_embeddings)
        html_embeddings = self.html_tag_embedding_projector(html_embeddings)
        xpath_embeddings = self.xpath_embedding_projector(xpath_embeddings)
        position_embeddings = self.posotion_embedding_projector(position_embeddings)
        if self.no_text_information:  # 11/20
            #assert args.use_xpath_embedding
            base_embeddings = xpath_embeddings * weights[1]
        else:
            base_embeddings = para_embeddings * weights[0]
            if self.xpath_embedding_dim > 0:
                base_embeddings = base_embeddings + xpath_embeddings * weights[1]
        if self.html_tag_embedding_dim > 0:
            base_embeddings = base_embeddings + html_embeddings * weights[2]
        if self.posotion_embedding_dim > 0:
            base_embeddings = base_embeddings + position_embeddings * weights[3]

        return base_embeddings

    def prepare_aggregation_method(self, fixed_final_hidden_dim=768):
        aggregation_method = self.aggregation_method
    
        if aggregation_method == "concat":
            final_hidden_dim = self.hidden_dim + self.html_tag_embedding_dim + self.xpath_embedding_dim + self.posotion_embedding_dim
            self.projector = lambda x:x
            do_aggregation = self.do_aggregation_concate
            
        elif aggregation_method == "concat-project":
            final_hidden_dim = fixed_final_hidden_dim  # TODO
            self.projector = nn.Linear(self.hidden_dim + self.html_tag_embedding_dim + self.xpath_embedding_dim + self.posotion_embedding_dim, final_hidden_dim)
            do_aggregation = self.do_aggregation_concat_project
            
        else:
            final_hidden_dim = max(self.hidden_dim, self.html_tag_embedding_dim, self.xpath_embedding_dim, self.posotion_embedding_dim)
            self.hidden_projector = nn.Linear(self.hidden_dim, final_hidden_dim) if (self.hidden_dim > 0 and self.hidden_dim != final_hidden_dim) else lambda x:x
            self.html_tag_embedding_projector = nn.Linear(self.html_tag_embedding_dim, final_hidden_dim) if (self.html_tag_embedding_dim > 0 and self.html_tag_embedding_dim != final_hidden_dim) else lambda x:x
            self.xpath_embedding_projector = nn.Linear(self.xpath_embedding_dim, final_hidden_dim) if (self.xpath_embedding_dim > 0 and self.xpath_embedding_dim != final_hidden_dim) else lambda x:x
            self.posotion_embedding_projector = nn.Linear(self.posotion_embedding_dim, final_hidden_dim) if (self.posotion_embedding_dim > 0 and self.posotion_embedding_dim != final_hidden_dim) else lambda x:x

            if aggregation_method == "add":
                do_aggregation = self.do_aggregation_add
                
    
            elif aggregation_method == "linear-combination-single-weight":
                self.aggregation_weights = torch.rand(4)  # 12/26
                if self.no_text_information:
                    self.aggregation_weights[0] = -1e8
                if self.xpath_embedding_dim == 0:
                    self.aggregation_weights[1] = -1e8
                if self.html_tag_embedding_dim == 0:
                    self.aggregation_weights[2] = -1e8
                if self.posotion_embedding_dim == 0:
                    self.aggregation_weights[3] = -1e8

                self.aggregation_weights = nn.parameter.Parameter(self.aggregation_weights)
                self.aggregation_softmax = torch.nn.Softmax(dim=-1)
                do_aggregation = self.do_aggregation_lc_simple
            
            elif aggregation_method == "linear-combination-complex-weight":
                # TODO
                # self.aggregation_weights = torch.rand(4)  # 12/26
                concate_hidden_dim = self.hidden_dim + self.html_tag_embedding_dim + self.xpath_embedding_dim + self.posotion_embedding_dim
                self.weight_linear = nn.Linear(concate_hidden_dim, 4)
                self.aggregation_softmax = torch.nn.Softmax(dim=-1)
                do_aggregation = self.do_aggregation_lc_complex

            
        return final_hidden_dim, do_aggregation
    

    def forward(self, input_ids, input_mask=None, edu_num=None, htmltag_ids=None, xpath_tag_ids=None, xpath_index=None, xpath_length=None, positions=None, **kwargs):
        """
        input_ids: (Batch, max_node_num, max_node_length), int ids
        input_mask: (Batch, max_node_num, max_node_length), 0/1, for indicating which tokens in each sentences are mask token, to guide the PLM attention mask matrix
        edu_num: (Batch), number of original nodes before padding in each document in the batch
        htmltag_ids: (Batch, max_node_num,), one node one html tag
        xpath_tag_ids: (Batch, max_node_num, max_xpath_length), one node a sequence
        xpath_index: (Batch, max_node_num, max_xpath_length), one node a sequence
        xpath_length: (Batch, max_node_num,), for indicating the original length of xpath sequence of each node
        positions: ???

        Since each node is computed locally, the only usage of edu_num is to guide global_encoder when calculate global node representation
        While since batch_size=1 by default, this is mostly not uesd.
        """
        if "sep_index_list" not in kwargs.keys():  # Local encoder 1/1
            batch_size, max_num_edu, _ = input_ids.shape
        else:  # all Concate encoder 1/1
            batch_size = input_ids.shape[0]
            max_num_edu = max([len(a) for a in kwargs["sep_index_list"]])

        para_embeddings = torch.tensor([[[]]*max_num_edu]*batch_size).to(input_ids.device)
        if not self.no_text_information:
            # para_embeddings = self.paragraph_encoder(input_ids, input_mask, )
            para_embeddings = self.paragraph_encoder(input_ids, input_mask, **kwargs)  # 12/26, 为适应ConcatDocumentEncoder

        #####use html tag and XPath features#####
        # batch_size, max_num_edu, _ = para_embeddings.shape
        html_embeddings = torch.tensor([[[]]*max_num_edu]*batch_size).to(para_embeddings.device)
        xpath_embeddings = torch.tensor([[[]]*max_num_edu]*batch_size).to(para_embeddings.device)
        position_embeddings = torch.tensor([[[]]*max_num_edu]*batch_size).to(para_embeddings.device)
        if self.use_html_embedding and self.html_tag_embedding is not None:
            html_embeddings = self.html_tag_embedding(htmltag_ids)
        if self.use_xpath_embedding and self.xpath_encoder is not None:
            xpath_embeddings = self.xpath_encoder(xpath_tag_ids, xpath_index, xpath_length)
        if self.use_position_embedding and self.relative_posotion_embedding is not None:
            position_embeddings = self.relative_posotion_embedding(positions)

        # if self.aggregation_method = "concat":
        #     para_embeddings = torch.cat((para_embeddings, html_embeddings, xpath_embeddings), dim=-1)
        # elif self.aggregation_method = "add":
        #     pass
        
        """para_embeddings = torch.cat((para_embeddings, html_embeddings, xpath_embeddings, position_embeddings), dim=-1)
        para_embeddings = self.projector(para_embeddings)"""
        para_embeddings = self.do_aggregation(para_embeddings, html_embeddings, xpath_embeddings, position_embeddings)  # 12/26

        
        if self.use_global_encoder:  # 1108
            if not edu_num:
                edu_num = [para_embeddings.shape[1]] # Not a batch implementation
            global_para_embeddings = self.global_encoder(para_embeddings, num_nodes=edu_num)
            if self.residual_connect:
                global_para_embeddings = para_embeddings + global_para_embeddings
            para_embeddings = global_para_embeddings
        
        return para_embeddings

from tqdm import tqdm
from typing import Dict, List
from collections import defaultdict

import os

import torch

# from utils import padding_matrix, self_adaption_length

def convert_figure(tokens):
    return "[图片]"


NODE_IDENTITY_dict = {0: "Title", 1: "Content", 2: "Site", 3: "Meta", 4: "Decorative", 5: None}
NODE_MODAL_dict = {0: "Title", 1: "Text", 2: "Figure", 3: "Figure&Title", 4: None}
# FATHER_RELATION_dict = {0: "Title", 1: "Attribute", 2: "Summary", 3: "Elaborate", 4: "Caption", 5: None}
# PREVIOUS_RELATION_dict = {0: "Continue", 1: "Break", 2: "Combine", 3: None}
FATHER_RELATION_dict = {1: "Elaboration", 2: "Topic&Key", 3: "Attribute", 4: "Explanation", 5: "Literature", 6:"Caption", 7:"Placeholder", 0: "NA"}
PREVIOUS_RELATION_dict = {1: "Narration", 2: "List", 3:"Parallel&Contrast", 4:"Topic_Transition", 5:"Break", 0: "NA"}
FATHER_RELATION_FINE_dict = {1:'Summary', 2:'Aggregation', 3:'Paraphrase', 4:'Partial',
5:'Tag', 6:'Topic', 7:'Content', 8:'Title', 9:'Quotation', 10:'Key-Value', 11:'Claim-Supporting',
12:'Question-Answer', 13:'Descriptive_Caption', 14:'Meta_Caption', 15:'Placeholder', 16:'Literature', 0:'NA'}
FATHER_RELATION_dict = {k-1:v for k, v in FATHER_RELATION_dict.items()}
PREVIOUS_RELATION_dict = {k-1:v for k, v in PREVIOUS_RELATION_dict.items()}
FATHER_RELATION_FINE_dict = {k-1:v for k, v in FATHER_RELATION_FINE_dict.items()}

new_FATHER_RELATION_dict = {1: "Elaboration", 2: "Topic&Key", 3: "Attribute", 4: "Explanation", 5: "Literature", 6:"Caption", 7:"Placeholder", 0: None}
new_PREVIOUS_RELATION_dict = {1: "Break", 2: "Continue", 0: None}
# new_FATHER_RELATION_grained_mapper = {0: "Title", 1: "Attribute", 2: "Summary", 3: "Elaborate", 4: "Caption", 5: None}
new_PREVIOUS_RELATION_fine_grained = {1: "Break", 2: "Narration", 3: "List", 4: "Parallel&Contrast", 5: "Topic_Transition", 6: "Background", 7: "Comment", 0: None}
new_PREVIOUS_RELATION_grained_mapper = {"Break":"Break", "Narration":"Continue", "List":"Continue", "Parallel&Contrast":"Continue", "Topic_Transition":"Continue", "Background":"Continue", "Comment":"Continue"}

# NODE_LEVEL_KEYS = ["Node_ID", "Content", "Node_Identity", "Node_modal", "Father", "Father_Relation", "Previous",
#                    "Previous_Relation", "Previous_Relation_Fine", "xpath", "htmltag"]
# NODE_LEVEL_KEYS = ["Node_ID", "Content", "Node_Identity", "Node_modal", "Father", "Father_Relation", "Previous",
#                    "Previous_Relation", "Previous_Relation_Fine",]
# NODE_LEVEL_KEYS = ["Node_ID", "Content", "Node_Identity", "Node_modal", "Father", "Father_Relation", "Father_Relation_Fine", "Previous",
#                    "Previous_Relation", "htmltag", "xpath", "real_xpath_string", "logical_level",]  # for data_split_new
NODE_LEVEL_KEYS = ["Node_ID", "Content", "Node_Identity", "Node_modal", "Father", "Father_Relation", "Father_Relation_Fine", "Previous",
                   "Previous_Relation", "htmltag", "xpath_global", "xpath_origin", "xpath_string", "logical_level",]  # for latestest data
INSTANCE_LEVEL_KEYS = ["id"]


def text_to_id(inputs: List, id_to_text_mapper: Dict):
    if type(inputs) != list:
        inputs = [inputs]
    text_to_id_mapper = {v: k for k, v in id_to_text_mapper.items()}
    return [text_to_id_mapper[text] if text in text_to_id_mapper.keys() else text_to_id_mapper[None] for text in inputs]


def add_node_to_instance(instance: Dict, node: Dict):
    assert set(node.keys()) == set(NODE_LEVEL_KEYS)
    for k, v in node.items():
        instance[k].append(v)

def self_adaption_length(length_list, total_max_length):
    assert total_max_length > 0
    if sum(length_list) <= total_max_length:
        return length_list
    while(sum(length_list) > total_max_length):
        length_list[length_list.index(max(length_list))] -= 1
    return length_list

class Vocab:
    def __init__(self, vocabulary=[], padding_value=0, mapping_dict=None):
        self._vocabulary = set(vocabulary)
        # self._size = len(vocabulary)
        self.token2id = mapping_dict or {t:i for i, t in enumerate(list(vocabulary))}
        self.id2token = {i:t for t, i in self.token2id.items()}
        
        self.embeddings = {}
        self.id2embeddings = {}
        self.padding_value = padding_value

    def load_embeddings(self, embeddings):
        """
        Update (overwrite) self.embeddings and self.id2embeddings from embeddings
        embeddings: Dict[token:str, torch.Tensor]
        """
        self.embeddings.update(embeddings)
        self.id2embeddings.update({self.token2id[token]: tensor for token, tensor in embeddings.items()})
        print("(class Vocab): vocab embeddings updated")
    
    def get_embeddings(self):
        """
        Get the embedding tensor which can be used to initialize an nn.Embedding layer
        """
        if not self.embeddings or not self.id2embeddings:
            print("(class Vocab): embeddings have not been initialized!")
            return
        embedding_lst = list(dict(sorted(self.id2embeddings, key=lambda x:x[0])).values())
        embedding_tensor = torch.concat([torch.unsqueeze(t,0) for t in embedding_lst], dim=0)
        return embedding_tensor
    
    def load_weight(self, weight):
        """
        Update (overwrite) self.embeddings and self.id2embeddings from nn.Embedding.weight
        weight: torch.Tensor(num_embeddings, embedding_dim)
        """
        assert len(weight.shape) == 2
        w = weight.detach().tolist()
        for i, t in enumerate(w):
            self.id2embeddings[i] = torch.tensor(t)
            self.embeddings[self.id2token[i]] = torch.tensor(t)


    @staticmethod
    def from_embedding_file(embedding_file):
        """
        initialization method, get a vocab object with self.token2id, self.id2token, self.embeddings, self.id2embeddings initialized
        """
        embeddings = torch.load(embedding_file)  # Dict
        vocab = Vocab(embeddings.keys())
        vocab.load_embeddings(embeddings)
        return vocab

    def save_embeddings(self, embedding_file):
        """
        save a file which can be used to initialize a vocab by `Vocab.from_embedding_file()`
        """
        assert self.embeddings is not None, "self.embeddings is None"
        if os.path.exist(embedding_file):
            print(f"(class Vocab): embedding_file path {embedding_file} is already exist, overwrite? y/[n]")
            c = input()
            if c != "y":
                return
        torch.save(self.embeddings, embedding_file)

    
    @property
    def size(self,):
        # return self._size
        assert len(self._vocabulary) == len(self.token2id.keys())
        assert len(self._vocabulary) == len(self.id2token.keys())
        return len(self._vocabulary)
    
    @staticmethod
    def from_text_lines(vocab_file):
        """
        initialization method, get a vocab object with self.token2id, self.id2token initialized, while self.embeddings, self.id2embeddings being {}
        """
        with open(vocab_file, "r", encoding="utf-8") as f:
            token_list = f.read().strip().split("\n")
        return Vocab(set(token_list))
    
    def save_vocab_to_text_lines(self, vocab_file):
        """
        save a file which can be used to initialize a vocab by `Vocab.from_text_lines()`
        """
        if os.path.exist(vocab_file):
            print(f"(class Vocab): vocab_file path {vocab_file} is already exist, overwrite? y/[n]")
            c = input()
            if c != "y":
                return
        with open(vocab_file, "w", encoding="utf-8") as f:
            d = dict(sorted(self.token2id.items(), key=lambda x:x[1],))
            f.write("\n".join(list(d.keys())))
    
    @staticmethod
    def from_token2id_mapper(token2id_mapper):
        """
        initialization method, get a vocab object with self.token2id, self.id2token initialized, while self.embeddings, self.id2embeddings being {}
        """
        vocab = list(token2id_mapper.keys())
        return Vocab(vocab, token2id_mapper)
    
    @staticmethod
    def from_token2id_mapper_file(vocab_file):
        """
        initialization method, get a vocab object with self.token2id, self.id2token initialized, while self.embeddings, self.id2embeddings being {}
        """
        with open(vocab_file, "r", encoding="utf-8") as f:
            token2id_mapper = eval(f.read().strip())
        return Vocab.from_token2id_mapper(token2id_mapper)
    
    def save_vocab_to_token2id_mapper(self, vocab_file):
        """
        save a file which can be used to initialize a vocab by `Vocab.from_token2id_mapper()`
        """
        if os.path.exists(vocab_file):
            print(f"(class Vocab): vocab_file path {vocab_file} is already exist, overwrite? y/[n]")
            c = input()
            if c != "y":
                return
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(str(self.token2id))
    
    def add_one_token(self, token):
        self._vocabulary.update([token])
        if token in self.token2id.keys():
            return
        else:
            idx = len(list(self.token2id.keys()))
            self.token2id[token] = idx
            self.id2token[idx] = token
    
    def add_batch_token(self, token_list):
        self._vocabulary.update(token_list)
        for token in token_list:
            if not token in self.token2id.keys():
                idx = len(list(self.token2id.keys()))
                self.token2id[token] = idx
                self.id2token[idx] = token




class WebDataProcessor:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        self.max_seq_length_for_concate = args.max_seq_length_for_concate
        self.text_encoder_type = args.text_encoder_type
        # self.label_field = labelfield
        # self.LABEL_TO_ID = labelfield.label2id
        self.text2id_mappers = {"Node_Identity": NODE_IDENTITY_dict,
                                "Node_modal": NODE_MODAL_dict,
                                "Father_Relation": FATHER_RELATION_dict,
                                "Previous_Relation": PREVIOUS_RELATION_dict,

                                "Father_Relation_Fine": FATHER_RELATION_FINE_dict,
                                }
        self.combine_before = args.combine_before
        self.max_paragraph_num = args.max_paragraph_num
        self.preprocess_figure_node = args.preprocess_figure_node
        self.discard_figure_node = args.discard_figure_node
        self.max_xpath_index_value = args.max_xpath_index_value
        self.html_tag_vocab = Vocab()
        self.xpath_tag_vocab = Vocab()

        self.xpath_key = "xpath_origin"

        add_string = self.get_data_process_version_tag(args)
        args.data_cache_key += add_string


    @staticmethod
    def get_data_process_version_tag(args):
        add_string = ""
        add_string += f"_{args.text_encoder_type}"
        if args.text_encoder_type == "concate":
            add_string += f"_{args.max_seq_length_for_concate}"
        if args.combine_before:
            add_string += "_combined"
        if args.discard_figure_node:
            add_string += "_nofigure"
        elif args.preprocess_figure_node:
            add_string += "_processfigure"
        add_string += f"_{args.max_seq_length}_{args.max_paragraph_num}"
        return add_string
    
    def get_html_vocab(self):
        return self.html_tag_vocab
    
    def get_xpath_vocab(self):
        return self.xpath_tag_vocab
    
    def set_html_vocab(self, vocab):
        self.html_tag_vocab = vocab
    
    def set_xpath_vocab(self, vocab):
        self.xpath_tag_vocab = vocab
    
    def get_features_from_dataset(self, dataset):
        print("Processing to get input features ...")
        features = []
        for instance in tqdm(dataset):
            feature = self.instance_to_features(
                instance)  # if not self.combine_before else self.instance_to_features_combine(instance)
            features.append(feature)
        return features

    def instance_to_features(self, instance):
        print(instance["id"])
        for k, v in instance.items():
            if k not in ["id"]:
                instance[k] = v[:self.max_paragraph_num]
        instance["Father"] = [i if i!="-1" else "NA" for i in instance["Father"]]
        instance["Previous"] = [i if i!="-1" else "NA" for i in instance["Previous"]]

        if self.combine_before:
            instance = self._do_combine(instance)
        
        if self.discard_figure_node:
            instance = self._discard_figure_node(instance)

        paragraphs = instance["Content"]
        paragraph_ids = []
        for tokens in paragraphs:
            if tokens.startswith("https://") or tokens.startswith("http://"):
                tokens = convert_figure(tokens) if self.args.preprocess_figure_node else tokens
            ## sents = self.tokenizer.tokenize(tokens)
            ## sents = sents[:self.max_seq_length]
            ## input_ids = self.tokenizer.convert_tokens_to_ids(sents)
            input_ids = self.tokenizer(tokens)["input_ids"][:self.max_seq_length]
            paragraph_ids.append(input_ids)
        # relation_id = self.LABEL_TO_ID[instance["relation_type"]]
        # node_identity_ids = instance["Node_Identity"]
        # node_modal_ids = instance["Node_modal"]
        # father_relation_ids = instance["Father_Relation"]
        # previous_relation_ids = instance["Previous_Relation"]
        instance["Father"] = [int(f) if f != "NA" else -1 for f in instance["Father"]]
        instance["Previous"] = [int(f) if f != "NA" else -1 for f in instance["Previous"]]
        # print(instance["Father"])

        instance.update({
            "input_ids": paragraph_ids, 
        })

        # instance = self.add_features(instance)

        for key, mapper in self.text2id_mappers.items():
            instance.update(
                {key + "_ids": text_to_id(instance[key], mapper)}
            )
        
        
        if self.xpath_key in instance.keys():
            instance[self.xpath_key] = [eval(f) for f in instance[self.xpath_key]]
            html_tag_from_xpath = [t[0] for xpath in instance[self.xpath_key] for t in xpath ]
            # self.html_tag_vocab.add_batch_token(instance["htmltag"] + html_tag_from_xpath)
            self.html_tag_vocab.add_batch_token(instance["htmltag"])
            self.xpath_tag_vocab.add_batch_token(html_tag_from_xpath)  
            
            instance.update({
                "htmltag_ids": text_to_id(instance["htmltag"], self.html_tag_vocab.id2token),
                # "xpath_ids": [ [(text_to_id(t[0], self.html_tag_vocab.id2token)[0], int(t[1])) for t in xpath] for xpath in instance["xpath"]],  # remember this [0]
                "xpath_tag_ids": [ [text_to_id(t[0], self.xpath_tag_vocab.id2token)[0] for t in xpath] for xpath in instance[self.xpath_key]],
                "xpath_index": [ [int(t[1]) if int(t[1])<self.max_xpath_index_value else self.max_xpath_index_value-1 for t in xpath] for xpath in instance[self.xpath_key]],
            })

        # instance["Previous_Relation_ids"] = [0 if id==3 else id for id in instance["Previous_Relation_ids"]]

        assert len(instance["Father"]) == len(paragraph_ids), instance["Father"]
        assert len(instance["Previous_Relation_ids"]) == len(paragraph_ids), instance["Previous_Relation"]

        if self.text_encoder_type == "concate":  # 12/26
            instance = self.add_concate_encoder_features(instance)

        instance = self.add_features(instance)
        
        return instance

    # @staticmethod
    def add_features(self, instance):
        """
        newly increased 10/18, for adding any custom features a specific model may take as inputs, from the original annotated data
        this method is expected to be reimplemented in sub-classes
        :param instance:
        :return:
        """
        return instance

    
    def _discard_figure_node(self, instance):
        # TODO
        return instance
    
    
    def _do_combine(self, instance):
        combined_instance = {k: [] for k in instance.keys()}
        combined_instance["id"] = instance["id"]
        id_projection_dict = {}
        current_node = {}
        for i in instance["Node_ID"]:

            i = int(i)
            # print(instance)
            node = {k: instance[k][i] for k in NODE_LEVEL_KEYS}

            if node["Previous_Relation"] != "Combine":

                node_id_before_combine = i
                node_id_after_combine = len(combined_instance["Node_ID"])
                id_projection_dict[node_id_before_combine] = node_id_after_combine

                for k, v in node.items():
                    # Creat a new current node
                    current_node[k] = v

                # Add current_node to combined_instance, therefore node_id_after_combine will += 1 (in later lines)
                current_node["Node_ID"] = node_id_after_combine
                # print(id_projection_dict)
                # print(current_node["Father"])
                current_node["Father"] = id_projection_dict[int(current_node["Father"])] if current_node[
                                                                                                "Father"] != "NA" else "NA"
                current_node["Previous"] = id_projection_dict[int(current_node["Previous"])] if current_node[
                                                                                                    "Previous"] != "NA" else "NA"
                add_node_to_instance(combined_instance, current_node)

            if node["Previous_Relation"] == "Combine":
                # DO NOT add current_node to combined_instance, therefore node_id_after_combine will not change,
                # that means the node will be combined into current_node and share the same node_id_after_combine

                # TODO: Way-2
                node_id_before_combine = i
                # print(instance)
                # print(instance["id"])
                node_id_after_combine = id_projection_dict[int(node["Previous"])]
                id_projection_dict[node_id_before_combine] = node_id_after_combine

                assert ((node["Father"] == "NA" and combined_instance["Father"][node_id_after_combine] == "NA") or
                        id_projection_dict[int(node["Father"])] == combined_instance["Father"][
                            node_id_after_combine]), (instance, node)
                assert node["Father_Relation"] == combined_instance["Father_Relation"][node_id_after_combine], (instance["id"], i)

                combined_instance["Content"][node_id_after_combine] += node["Content"]
                if node["Node_modal"] == "Text":
                    combined_instance["Node_modal"][node_id_after_combine] = "Text"
                # assert node["Father"] == current_node["Father"], (instance, node)
                # assert node["Father_Relation"] == current_node["Father_Relation"]
                # assert int(node["Previous"]) == i-1, (instance, node)

            # TODO: Way-1
            """node_id_before_combine = i
            node_id_after_combine = len(combined_instance["Node_ID"])
            id_projection_dict[node_id_before_combine] = node_id_after_combine"""
        return combined_instance

    def add_concate_encoder_features(self, instance):
        # model_name_or_path='hfl/chinese-xlnet-base'
        # 12/26
        paragraphs = instance["Content"]
        num_paragraph = len(paragraphs)
        # max_seq_length_per_paragraph = (self.max_seq_length_for_concate // num_paragraph) - 1
        max_seq_length_per_paragraph = self_adaption_length([len(self.tokenizer.tokenize(tokens)) for tokens in paragraphs], self.max_seq_length_for_concate)
        max_seq_length_per_paragraph = [i-1 for i in max_seq_length_per_paragraph]
        
        # paragraph_ids = []
        tokenized_texts = []
        for i, tokens in enumerate(paragraphs):
            if tokens.startswith("https://") or tokens.startswith("http://"):
                tokens = convert_figure(tokens) if self.args.preprocess_figure_node else tokens
            ## sents = self.tokenizer.tokenize(tokens)
            ## sents = sents[:self.max_seq_length]
            ## input_ids = self.tokenizer.convert_tokens_to_ids(sents)

            text_tokens = self.tokenizer.tokenize(tokens)[:max_seq_length_per_paragraph[i]]
            # input_ids = self.tokenizer(tokens)["input_ids"][:self.max_seq_length]  # 1210
            text_tokens = ['<sep>'] + text_tokens

            tokenized_texts.append(text_tokens)
        
        tokenized_document = []  # ['<sep>', edu1token1, edu1token2, ..., '<sep>', edu2token1, ...]
        for tokenized_paragrah in tokenized_texts:
            tokenized_document.extend(tokenized_paragrah)
        tokenized_document = tokenized_document + ['<cls>']

        temp_sep_index_list = []
        for index, token in enumerate(tokenized_document):
            if token == '<sep>':
                temp_sep_index_list.append(index)
        
        tokenized_document = self.tokenizer.convert_tokens_to_ids(tokenized_document)
        """total_tokens = torch.LongTensor(total_tokens)"""

        instance.update(
            {
                "tokenized_document": tokenized_document,  # List[int], (num_edu), indicating the number of tokens in each EDU in the instance
                "sep_index_list": temp_sep_index_list,  # # int, the number of EDUs in this instance
                
            }
        )
        return instance



class ConcateEncoderProcessor(WebDataProcessor):
    # That is just similar with SSAProcessor
    def __init__(self, args, tokenizer):
        super(ConcateEncoderProcessor, self).__init__(args, tokenizer)
        self.max_seq_length_for_concate = 1024

    # @staticmethod
    def add_features(self, instance,):
        
        paragraphs = instance["Content"]
        num_paragraph = len(paragraphs)
        max_seq_length_per_paragraph = (self.max_seq_length_for_concate // num_paragraph) - 1
        
        # paragraph_ids = []
        tokenized_texts = []
        for tokens in paragraphs:
            if tokens.startswith("https://") or tokens.startswith("http://"):
                tokens = convert_figure(tokens) if self.args.preprocess_figure_node else tokens
            ## sents = self.tokenizer.tokenize(tokens)
            ## sents = sents[:self.max_seq_length]
            ## input_ids = self.tokenizer.convert_tokens_to_ids(sents)

            text_tokens = self.tokenizer.tokenize(tokens)[:max_seq_length_per_paragraph]
            # input_ids = self.tokenizer(tokens)["input_ids"][:self.max_seq_length]  # 1210
            text_tokens = ['<sep>'] + text_tokens

            tokenized_texts.append(text_tokens)
        
        tokenized_document = []  # ['<sep>', edu1token1, edu1token2, ..., '<sep>', edu2token1, ...]
        for tokenized_paragrah in tokenized_texts:
            tokenized_document.extend(tokenized_paragrah)
        tokenized_document = ['<cls>'] + tokenized_document


        temp_sep_index_list = []
        for index, token in enumerate(tokenized_document):
            if token == '<sep>':
                temp_sep_index_list.append(index)
        
        tokenized_document = self.tokenizer.convert_tokens_to_ids(tokenized_document)
        """total_tokens = torch.LongTensor(total_tokens)"""

        instance.update(
            {
                "tokenized_document": tokenized_document,  # List[int], (num_edu), indicating the number of tokens in each EDU in the instance
                "sep_index_list": temp_sep_index_list,  # # int, the number of EDUs in this instance
                
            }
        )
        return instance
        


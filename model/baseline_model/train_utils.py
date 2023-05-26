import os
import torch
from transformers import AutoConfig, AutoTokenizer

from .model2 import BaselineModel
from .structured_global import BaselineModelStructuredGlobal
from .args import parser as baseline_parser
from processor import WebDataProcessor
from .utils2 import collate_fn

from module import BaseNodeEncoder

from train_utils import BaseTrainEnv

class BaselineTrainEnv(BaseTrainEnv):

    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group('baseline')

        """group.add_argument('-alpha', dest='alpha', default=1., type=float, help='Alpha weight for Bag Loss')
        group.add_argument('-margin', dest='margin', default=0.75, type=float, help='Margin for Boundary Loss')
        group.add_argument('-neg-weight', dest='neg_weight', default=2.0, type=float, help='Negative Weight')
        group.add_argument('-detect-win', dest='detect_win', default=-1, type=int, help='Detection Window Size')
        group.add_argument('-arn-dropout', dest='arn_dropout', default=0.3, type=float, help='Dropout Rate')
        group.add_argument('-boundary-conv-size', dest='boundary_conv_size', default=2, type=int,
                           help='Boundary Conv Size')"""


        # parser.add_argument("--golden_parent_when_evaluate", default=False, action="store_true")


        group.add_argument("--loss_type", default="ce", type=str, choices=["margin", "ce"])  # there seems some problem with margin loss, use ce always. 1106
        group.add_argument("--alpha", default=0.5, type=float, help="hyperparameter for joint loss.")
        
        # parser.add_argument("--layer_type", default="bilinear", type=str,
        #                     choices=["linear", "bilinear", "attention", "mixed", "new"])  # deprecated 11/19
        # parser.add_argument("--additional_encoder", default=False, action="store_true")  # deprecated 11/18
        # parser.add_argument("--additional_encoder_type", default="transformer",
        #                     choices=["transformer", "lstm", "gru", "linear"])  # deprecated 11/18
        
        # not used
        group.add_argument("--position_encoding_type", default=None, type=str,
                            choices=["random", "one-hot", "cosine", None])
        group.add_argument("--position_encoding_dim", default=512, type=int, )
        group.add_argument("--position_combine_method", default=None, type=str,
                            choices=["add", "concate", "add-linear", "concate-linear", None])
        
        # inject head-tail relative position information when predict parent node
        '''group.add_argument("--relative_position_encoding_type", default="random", type=str,
                            choices=["random", "one-hot", "cosine", None])
        group.add_argument("--relative_position_encoding_dim", default=200, type=int, )
        group.add_argument("--relative_position_combine_method", default="bilinear", type=str,
                            choices=["bilinear", "linear", "mlp", "trilinear"])
        group.add_argument("--bilinear_combine", default="tail", type=str, choices=["head", "tail", "both"])'''  # 12/31
        
        # use history structure information during auto-regressive parent node decoding
        group.add_argument("--global_sequence_module", default=None, type=str, choices=["lstm", None])
        
        group.add_argument("--relation_embedding_dim", default=256, type=int, help="used in deepseq model, the dimension of historical relation embedding")
        group.add_argument("--structured_representation_dim", default=512, type=int, help="used in deepseq model, the dimension of structured global representation")
        

    @staticmethod
    def prepare_tokenizer(args):
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        )
        return tokenizer

    @staticmethod
    def prepare_model(args, tokenizer, data_processor):
        config = AutoConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            # num_labels=args.num_class,
        )
        config.gradient_checkpointing = True  # TODO

        # model = BaselineModel(args, config)
        node_encoder = BaseNodeEncoder(args, config, data_processor)  # 11/19
        model = BaselineModel(args, config, node_encoder)  # 11/19

        if args.test_only:
            # load_path = os.path.join(args.model_path, f"checkpoint_{args.test_checkpoint_id}.pkl")
            load_path = args.test_checkpoint_dir
            print(f"Loading NN model pretrained checkpoint from {load_path} ...")
            model.load_state_dict(torch.load(load_path))
        model.to(args.device)
        return model

    '''@staticmethod
    def get_param_groups(args, model):
        """param_groups = [{"params": model.paragraph_encoder.parameters(), "lr": args.transformer_learning_rate}]
        param_groups += [{"params": filter(lambda p: id(p) not in list(map(id, model.paragraph_encoder.parameters())),
                                           model.parameters()),
                          "lr": args.learning_rate}]"""
        # 1210
        transformer_parameters = model.paragraph_encoder.paragraph_encoder.parameters() if hasattr(model.paragraph_encoder, "paragraph_encoder") else model.paragraph_encoder.parameters()
        
        print(list(map(id, transformer_parameters)))
        print(list(map(id, model.paragraph_encoder.paragraph_encoder.parameters())))
        param_groups = [{"params": transformer_parameters,
         "lr": args.transformer_learning_rate}]  # paragraph_encoder
        param_groups += [{"params": filter(lambda p: id(p) not in list(map(id, transformer_parameters)),
                                           model.parameters()),
                          "lr": args.learning_rate}]
        return param_groups'''

    @staticmethod
    def prepare_argparser():
        return baseline_parser

    @staticmethod
    def prepare_dataprocessor(args, tokenizer):
        processor = WebDataProcessor(args, tokenizer)
        return processor

    @staticmethod
    def get_train_collate_fn(data_processor=None):
        return collate_fn

    @staticmethod
    def get_test_collate_fn(data_processor=None):
        return BaselineTrainEnv.get_train_collate_fn()


class DeepSeqTrainEnv(BaselineTrainEnv):
    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group('deepseq')
        # parser.add_argument("--golden_parent_when_evaluate", default=False, action="store_true")


        group.add_argument("--loss_type", default="ce", type=str, choices=["margin", "ce"])  # there seems some problem with margin loss, use ce always. 1106
        group.add_argument("--alpha", default=0.5, type=float, help="hyperparameter for joint loss.")
        
        # parser.add_argument("--layer_type", default="bilinear", type=str,
        #                     choices=["linear", "bilinear", "attention", "mixed", "new"])  # deprecated 11/19
        # parser.add_argument("--additional_encoder", default=False, action="store_true")  # deprecated 11/18
        # parser.add_argument("--additional_encoder_type", default="transformer",
        #                     choices=["transformer", "lstm", "gru", "linear"])  # deprecated 11/18
        
        # not used
        group.add_argument("--position_encoding_type", default=None, type=str,
                            choices=["random", "one-hot", "cosine", None])
        group.add_argument("--position_encoding_dim", default=512, type=int, )
        group.add_argument("--position_combine_method", default=None, type=str,
                            choices=["add", "concate", "add-linear", "concate-linear", None])
        
        # inject head-tail relative position information when predict parent node
        '''group.add_argument("--relative_position_encoding_type", default="random", type=str,
                            choices=["random", "one-hot", "cosine", None])
        group.add_argument("--relative_position_encoding_dim", default=200, type=int, )
        group.add_argument("--relative_position_combine_method", default="bilinear", type=str,
                            choices=["bilinear", "linear", "mlp", "trilinear"])
        group.add_argument("--bilinear_combine", default="tail", type=str, choices=["head", "tail", "both"])'''  # 12/31
        
        # use history structure information during auto-regressive parent node decoding
        group.add_argument("--global_sequence_module", default=None, type=str, choices=["lstm", None])
        
        group.add_argument("--relation_embedding_dim", default=256, type=int, help="used in deepseq model, the dimension of historical relation embedding")
        group.add_argument("--structured_representation_dim", default=512, type=int, help="used in deepseq model, the dimension of structured global representation")


    @staticmethod
    def prepare_model(args, tokenizer, data_processor):
        config = AutoConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            # num_labels=args.num_class,
        )
        config.gradient_checkpointing = True  # TODO

        # model = BaselineModel(args, config)
        node_encoder = BaseNodeEncoder(args, config, data_processor)  # 11/19
        model = BaselineModelStructuredGlobal(args, config, node_encoder)  # 11/19 # 12/2

        if args.test_only:
            # load_path = os.path.join(args.model_path, f"checkpoint_{args.test_checkpoint_id}.pkl")
            load_path = args.test_checkpoint_dir
            print(f"Loading NN model pretrained checkpoint from {load_path} ...")
            model.load_state_dict(torch.load(load_path))
        model.to(args.device)
        return model
import os
import torch
from transformers import AutoConfig, AutoTokenizer

from .putorskip_processor import PutOrSkipProcessor
from .args import parser as put_or_skip_parser
# from .discriminator import PutOrskipModel
from .discriminator2 import PutOrskipModel
# from .discriminator_new import PutOrskipModel as PutOrskipModelNew
from .utils import putorskip_collate_fn

# from module import prepare_html_tag_embedding, prepare_xpath_encoder
from module import BaseNodeEncoder
from train_utils import BaseTrainEnv


class POSTrainEnv(BaseTrainEnv):
    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group('pos')
        # group.add_argument("--golden_parent_when_evaluate", default=False, action="store_true")

        group.add_argument("--loss_type", default="margin", type=str, choices=["margin", "ce"])
        group.add_argument("--layer_type", default="bilinear", type=str,
                           choices=["linear", "bilinear", "attention", "mixed", "new"])
        group.add_argument("--additional_encoder", default=False, action="store_true")
        group.add_argument("--additional_encoder_type", default="transformer",
                            choices=["transformer", "lstm", "gru", "linear"])
        # parser.add_argument("--preprocess_figure_node", default=False, action="store_true",
        #                     help="whether to convert url of all figures to a unified linguistic utterance, e.g. 'tu-pian', during data processing")
        
        # parser.add_argument("--max_paragraph_num", default=200, type=int, )
        '''group.add_argument("--position_encoding_type", default=None, type=str,
                           choices=["random", "one-hot", "cosine", None])
        group.add_argument("--position_encoding_dim", default=512, type=int, )
        group.add_argument("--position_combine_method", default=None, type=str,
                           choices=["add", "concate", "add-linear", "concate-linear", None])
        group.add_argument("--relative_position_encoding_type", default="random", type=str,
                            choices=["random", "one-hot", "cosine", None])
        # parser.add_argument("--relative_position_encoding_dim", default=200, type=int, )
        group.add_argument("--relative_position_combine_method", default="bilinear", type=str,
                            choices=["bilinear", "linear", "mlp", "trilinear"])
        group.add_argument("--bilinear_combine", default="tail", type=str, choices=["head", "tail", "both"])'''
        group.add_argument("--global_sequence_module", default=None, type=str, choices=["lstm", None])
        
        
        group.add_argument("--alpha", default=0.5, type=float,
                            help="hyperparameter for joint loss.")
        # parser.add_argument("--scheduler", default="linear", type=str, choices=["constant", "warmup", "linear", "cosine"],
        #                     help="type of learning rate scheduler.")
        
        # parser.add_argument("--seed", type=int, default=42,
        #                     help="random seed for initialization")
        
        
        # put-or-skip model specific arguments
        group.add_argument("--hidden_dim", default=768, type=int)
        # parser.add_argument("--out_dim", default=6, type=int, help="relation type number")
        
        group.add_argument("--train_possible_position", default="all", choices=["all", "right", "right+maintitle"], type=str)
        group.add_argument("--decode_possible_position", default="all", choices=["all", "right", "right+maintitle"], type=str)
        
        group.add_argument("--negative_sampling_ratio", default=0.02, type=float)

        group.add_argument('--unified_previous_classifier', action="store_true",)
        
        
        '''group.add_argument("--pos_use_relative_position", default=True, action="store_true")
        group.add_argument("--pos_relative_position_encoding_dim", default=128, type=int)'''


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


        # html_tag_embedding = prepare_html_tag_embedding(args, html_tag_vocab) if args.use_html_embedding else None
        # xpath_encoder = prepare_xpath_encoder(args, html_tag_vocab) if args.use_xpath_embedding else None
        
        # model = PutOrskipModel(args, config, html_tag_embedding=html_tag_embedding, xpath_encoder=xpath_encoder)
        node_encoder = BaseNodeEncoder(args, config, data_processor)
        model = PutOrskipModel(args, config, node_encoder)

        if args.test_only:
            # load_path = os.path.join(args.model_path, f"checkpoint_{args.test_checkpoint_id}.pkl")
            load_path = args.test_checkpoint_dir
            print(f"Loading NN model pretrained checkpoint from {load_path} ...")
            model.load_state_dict(torch.load(load_path))
        model.to(args.device)
        return model

    @staticmethod
    def prepare_argparser():
        return put_or_skip_parser

    @staticmethod
    def prepare_dataprocessor(args, tokenizer):
        processor = PutOrSkipProcessor(args, tokenizer)
        return processor

    @staticmethod
    def get_train_collate_fn(data_processor=None):
        return putorskip_collate_fn

    @staticmethod
    def get_test_collate_fn(data_processor=None):
        return POSTrainEnv.get_train_collate_fn()

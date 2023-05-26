import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--golden_parent_when_evaluate", default=False, action="store_true")

parser.add_argument("--loss_type", default="margin", type=str, choices=["margin", "ce"])
parser.add_argument("--layer_type", default="bilinear", type=str,
                    choices=["linear", "bilinear", "attention", "mixed", "new"])
parser.add_argument("--additional_encoder", default=False, action="store_true")
parser.add_argument("--additional_encoder_type", default="transformer",
                    choices=["transformer", "lstm", "gru", "linear"])
# parser.add_argument("--preprocess_figure_node", default=False, action="store_true",
#                     help="whether to convert url of all figures to a unified linguistic utterance, e.g. 'tu-pian', during data processing")

# parser.add_argument("--max_paragraph_num", default=200, type=int, )
parser.add_argument("--position_encoding_type", default=None, type=str,
                    choices=["random", "one-hot", "cosine", None])
parser.add_argument("--position_encoding_dim", default=512, type=int, )
parser.add_argument("--position_combine_method", default=None, type=str,
                    choices=["add", "concate", "add-linear", "concate-linear", None])
parser.add_argument("--relative_position_encoding_type", default="random", type=str,
                    choices=["random", "one-hot", "cosine", None])
# parser.add_argument("--relative_position_encoding_dim", default=200, type=int, )
parser.add_argument("--relative_position_combine_method", default="bilinear", type=str,
                    choices=["bilinear", "linear", "mlp", "trilinear"])
parser.add_argument("--bilinear_combine", default="tail", type=str, choices=["head", "tail", "both"])
parser.add_argument("--global_sequence_module", default=None, type=str, choices=["lstm", None])


parser.add_argument("--alpha", default=0.5, type=float,
                    help="hyperparameter for joint loss.")


# put-or-skip model specific arguments
parser.add_argument("--hidden_dim", default=512, type=int)
# parser.add_argument("--out_dim", default=6, type=int, help="relation type number")


parser.add_argument("--train_possible_position", default="all", choices=["all", "right", "right+maintitle"], type=str)
parser.add_argument("--decode_possible_position", default="all", choices=["all", "right", "right+maintitle"], type=str)

parser.add_argument("--negative_sampling_ratio", default=0.1, type=float)


parser.add_argument("--pos_use_relative_position", default=True, action="store_true")
parser.add_argument("--pos_relative_position_encoding_dim", default=128, type=int)
import argparse

parser = argparse.ArgumentParser()


parser.add_argument("--golden_parent_when_evaluate", default=False, action="store_true")


parser.add_argument("--loss_type", default="ce", type=str, choices=["margin", "ce"])  # there seems some problem with margin loss, use ce always.
parser.add_argument("--alpha", default=0.5, type=float, help="hyperparameter for joint loss.")

# not used
parser.add_argument("--position_encoding_type", default=None, type=str,
                    choices=["random", "one-hot", "cosine", None])
parser.add_argument("--position_encoding_dim", default=512, type=int, )
parser.add_argument("--position_combine_method", default=None, type=str,
                    choices=["add", "concate", "add-linear", "concate-linear", None])

# inject head-tail relative position information when predict parent node
parser.add_argument("--relative_position_encoding_type", default="random", type=str,
                    choices=["random", "one-hot", "cosine", None])
parser.add_argument("--relative_position_encoding_dim", default=200, type=int, )
parser.add_argument("--relative_position_combine_method", default="bilinear", type=str,
                    choices=["bilinear", "linear", "mlp", "trilinear"])
parser.add_argument("--bilinear_combine", default="tail", type=str, choices=["head", "tail", "both"])

# use history structure information during auto-regressive parent node decoding
parser.add_argument("--global_sequence_module", default=None, type=str, choices=["lstm", None])

parser.add_argument("--relation_embedding_dim", default=256, type=int, help="used in deepseq model, the dimension of historical relation embedding")
parser.add_argument("--structured_representation_dim", default=512, type=int, help="used in deepseq model, the dimension of structured global representation")

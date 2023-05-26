import argparse

parser = argparse.ArgumentParser()

# parser.add_argument('--train_file', type=str)
# parser.add_argument('--eval_file', type=str)
# parser.add_argument('--test_file', type=str)
# parser.add_argument('--dataset_dir', type=str, default='dataset')

# parser.add_argument('--glove_vocab_path', type=str)  ### use transformers vocab directly
# parser.add_argument('--model_name_or_path', type=str, default='xlnet-base-cased')
# parser.add_argument('--max_vocab_size', type=int, default=1000)  ### use transformers vocab directly
# parser.add_argument('--remake_dataset', action="store_true")
# parser.add_argument('--remake_tokenizer', action="store_true")  ### use transformers vocab directly
"""parser.add_argument('--max_edu_dist', type=int, default=20)  # √"""
parser.add_argument('--max_edu_dist', type=int, default=999)  # √
# parser.add_argument('--glove_embedding_size', type=int, default=100)  ### use transformers vocab directly

parser.add_argument('--path_hidden_size', type=int, default=384)  # √
parser.add_argument('--hidden_size', type=int, default=768)  # ×
parser.add_argument('--num_layers', type=int, default=3)  # √
parser.add_argument('--num_heads', type=int, default=4)  # √
parser.add_argument('--dropout', type=float, default=0.5)  # √
parser.add_argument('--attention_dropout_DCA', type=float, default=0.1)  # This parameter has not been used in source code
parser.add_argument('--speaker', action='store_true')  # This parameter has not been used in source code
parser.add_argument('--valid_dist', type=int, default=10)  # √
# parser.add_argument('--learning_rate', type=float, default=3e-4)
# parser.add_argument('--pretrained_model_learning_rate', type=float, default=1e-5)
# parser.add_argument('--epoches', type=int, default=10)
# parser.add_argument('--pool_size', type=int, default=1)
# parser.add_argument('--eval_pool_size', type=int, default=1)
# parser.add_argument('--batch_size', type=int, default=1)
# parser.add_argument('--gamma', type=float, default=0)
# parser.add_argument('--save_model', action='store_true')
# parser.add_argument('--model_path', type=str, default='student_model.pt')
# parser.add_argument('--do_train', action="store_true")
# parser.add_argument('--do_eval', action="store_true")
# parser.add_argument('--report_step', type=int, default=50)

# parser.add_argument('--load_model', action='store_true')

# parser.add_argument('--early_stop', type=int, default=5)
########parser.add_argument('--text_max_sep_len', type=int, default= 31)
########parser.add_argument('--total_seq_len', type=int, default= 512)
# parser.add_argument('--seed', type=int, default= 512)
parser.add_argument('--decoder_input_size', type=int, default= 384)  # √
parser.add_argument('--decoder_hidden_size', type=int, default= 384)  # √
parser.add_argument('--classes_label', type=int, default= 17)  # ×
parser.add_argument('--transition_weight', type=int, default= 1, help="transition loss weight in multi-task loss")  # √
parser.add_argument('--graph_weight', type=int, default= 1, help="graph loss weight in multi-task loss")  # √
parser.add_argument('--add_norm', type=bool, default= True)  # √

parser.add_argument('--dagcn_embedding_dims', type=int, default=8, help="d prime in the paper")  # √
parser.add_argument('--dagcn_valid_dist', type=int, default=5, help="the embedding space dimension for distance embedding in distance-aware graph module")  # √

parser.add_argument('--split_hidden_size', type=int, default=64)  # √
parser.add_argument('--biaffine_hidden_size', type=int, default=128)  # √

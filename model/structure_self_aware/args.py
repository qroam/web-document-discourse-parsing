import argparse

parser = argparse.ArgumentParser()



# dataset
# parser.add_argument('--train_file', type=str)
# parser.add_argument('--eval_file', type=str)
# parser.add_argument('--test_file', type=str)
# parser.add_argument('--dataset_dir', type=str, default='data_split')

# TODO: wordvec is current invalid
# parser.add_argument('--glove_vocab_path', type=str)  # /data/share/model/word2vec/GitHub_chinese_word_vectors/sgns.financial.bigram-char
# # /data/share/model/word2vec/GitHub_chinese_word_vectors/merge_sgns_bigram_char300.txt
# parser.add_argument('--max_vocab_size', type=int, default=1000)
# parser.add_argument('--remake_dataset', action="store_true")
# parser.add_argument('--remake_tokenizer', action="store_true")

# parser.add_argument('--max_edu_dist', type=int, default=20)  # TODO # mask = get_mask(node_num=edu_nums + 1, max_edu_dist=args.max_edu_dist).cuda() {main:3}
parser.add_argument('--max_edu_dist', type=int, default=999)


# model
# parser.add_argument('--glove_embedding_size', type=int, default=300)  # TODO: wordvec is current invalid

# parser.add_argument('--hidden_size', type=int, default=256)
# parser.add_argument('--path_hidden_size', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=768,
    help="since residual connection is used, this size should be equal with encoder hidden dim")
parser.add_argument('--path_hidden_size', type=int, default=256)

parser.add_argument('--num_layers', type=int, default=3)  # layer number of GNN
parser.add_argument('--num_heads', type=int, default=4)  # head number of GNN, only used in StructureAwareAttention as a parameter
# parser.add_argument('--num_layers', type=int, default=1)  # layer number of GNN
# parser.add_argument('--num_heads', type=int, default=1)  # head number of GNN, only used in StructureAwareAttention as a parameter

parser.add_argument('--dropout', type=float, default=0.5)

# parser.add_argument('--speaker', action='store_true')
parser.add_argument('--valid_dist', type=int, default=10)  # only used in PathEmbedding

# train
# parser.add_argument('--learning_rate', type=float, default=0.1)
# parser.add_argument('--min_lr', type=float, default=0.01)
# parser.add_argument('--epoches', type=int, default=10)
# parser.add_argument('--pool_size', type=int, default=1)
# parser.add_argument('--eval_pool_size', type=int, default=1)
# parser.add_argument('--batch_size', type=int, default=1)
# parser.add_argument('--gamma', type=float, default=0.998)
# parser.add_argument('--ratio', type=float, default=1.0)

# save
# parser.add_argument('--save_model', action='store_true')
# # parser.add_argument('--model_path', type=str, default='student_model.pt')
# # parser.add_argument('--teacher_model_path', type=str, default='teacher_model.pt')
# parser.add_argument('--overwrite', action="store_true")

# other option
# parser.add_argument('--do_train', action="store_true")
# parser.add_argument('--do_eval', action="store_true")
# parser.add_argument('--report_step', type=int, default=50)
# parser.add_argument('--load_model', action='store_true')
# parser.add_argument('--early_stop', type=int, default=5)
# TODO: implementation
parser.add_argument('--task', type=str, default="student", choices=["teacher", "student", "distill"])
parser.add_argument('--classify_loss', action='store_true')
parser.add_argument('--classify_ratio', type=float, default=0.2)
parser.add_argument('--distill_ratio', type=float, default=3.)

# parser.add_argument('--use_negative_loss', type=bool, default=False)
parser.add_argument('--use_negative_loss', action="store_true",)  # 12/15
parser.add_argument('--negative_loss_weight', type=float, default=0.2)  # 12/15

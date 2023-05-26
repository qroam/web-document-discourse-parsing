# -*- coding: utf-8 -*-
import argparse
import os
import pickle
import time
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from utils import set_seed, prepare_optimizer, prepare_scheduler, difference_between_list, add_common_arguments
from utils import create_log, get_localtime, get_time_lag
from utils import flatten_list
from utils import DocumentRecorder
from utils import prepare_all_argparsers

from processor import WebDataProcessor
# from put_or_skip.putorskip_processor import PutOrSkipProcessor
from dataset import WebDataset

# from baseline_model.model import BaselineModel
# from put_or_skip.discriminator import PutOrskipModel
# from put_or_skip.discriminator_new import PutOrskipModel
# from baseline_model.utils import collate_fn
# from put_or_skip.utils import putorskip_collate_fn
from metrics import ListAccuracyMetric, SubdiscoursePRFMetric, MultipleClassificationAndListAccuracyMetric, MaskedListAccuracyMetric, MultipleConditionListAccuracyMetric
# from args import parser as baseline_parser
# from put_or_skip.args import parser as put_or_skip_parser
# from structure_self_aware.args import parser as ssa_parser
# from structure_self_aware.processor import SSAProcessor
# from structure_self_aware.utils import eval_collate_fn as ssa_collate_fn
# from structure_self_aware.model import StudentModel as SSAStudentModel
from processor import FATHER_RELATION_dict, PREVIOUS_RELATION_dict, Vocab

# from baseline_model.train_utils import BaselineTrainEnv
# from put_or_skip.train_utils import POSTrainEnv
# from structure_self_aware.train_utils import SSATrainEnv
from model import BaselineTrainEnv, SSATrainEnv, POSTrainEnv, DeepSeqTrainEnv, DAMTTrainEnv#, SDDPTrainEnv

# from processor import FATHER_RELATION_dict

# from data_backup import build_vocab_from_dataset, PairDataset
"""parser_dict = {
    "baseline": baseline_parser,
    "putorskip": put_or_skip_parser,
    "ssa": ssa_parser,
    "deepseq": None,
}
processor_dict = {
    "baseline": WebDataProcessor,
    "putorskip": PutOrSkipProcessor,
    "ssa": SSAProcessor,
    "deepseq": None,
}
collate_fn_dict = {
    "baseline": collate_fn,
    "putorskip": putorskip_collate_fn,
    "ssa": ssa_collate_fn,
    "deepseq": None,
}
model_dict = {
    "baseline": BaselineModel,
    "putorskip": PutOrskipModel,
    "ssa": SSAStudentModel,
    "deepseq": None,
}"""

train_env_dict = {
    "baseline": BaselineTrainEnv,
    "putorskip": POSTrainEnv,
    "ssa": SSATrainEnv,
    "deepseq": DeepSeqTrainEnv,
    "damt": DAMTTrainEnv,
    # "sddp": SDDPTrainEnv,
    # "transition": None,
}

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def train(args, train_env, model, train_features, benchmarks, train_collate_fn, eval_collate_fn):
    with open(args.log_dir, "a+", encoding="utf-8") as metric_file:
        metric_file.write("\n" + "=" * 20 + "\n")
        metric_file.write(get_localtime() + "\n")
        # metric_file.write(f"train_batch_size = {args.train_batch_size} \
        # gradient_accumulation_steps = {args.gradient_accumulation_steps} \
        # learning_rate = {args.learning_rate} \
        # loss_type = {args.loss_type} \
        # alpha = {args.alpha}" + "\n")
        for arg_name in list(vars(args).keys()):
            metric_file.write(f"{arg_name} = {vars(args)[arg_name]}" + "\n")

    train_dataloader = DataLoader(train_features, batch_size=args.train_batch_size, shuffle=True, collate_fn=train_collate_fn,
                                  drop_last=False)
    print("train_dataloader prepared")

    total_steps = int(len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    print('Total steps: {}'.format(total_steps))
    print('Warmup steps: {}'.format(warmup_steps))

    # scaler = GradScaler()  # TODO "torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling."

    """params = [{"params": model.paragraph_encoder.parameters(),
               "lr": args.transformer_learning_rate}]
    params += [{"params": filter(lambda p: id(p) not in list(map(id, model.paragraph_encoder.parameters())),
                                 model.parameters()),
                "lr": args.learning_rate}]"""
    # TODO: param_groups = model_utils.prepare_params(args, model)
    # optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    # optimizer = AdamW(train_env.get_param_groups(args, model), eps=args.adam_epsilon)
    optimizer = prepare_optimizer(args, train_env.get_param_groups(args, model))
    # TODO: optimizer = utils.prepare_optimizer(args, param_groups)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scheduler = prepare_scheduler(args.scheduler, optimizer, warmup_steps, total_steps)

    print(f"model size: {sum(param.numel() for param in model.parameters())}")

    evaluation_metrics = {}
    for tag, features in benchmarks:
        evaluation_metrics[tag] = {
            # "parent_acc": ListAccuracyMetric(metric_name="parent_acc", dataset_name=tag),  # parent node UAS
            "parent_uas": MaskedListAccuracyMetric(metric_name="parent_uas", dataset_name=tag),  # parent node UAS
            # "previous_acc": ListAccuracyMetric(metric_name="previous_acc", dataset_name=tag),  # previous node UAS
            "parent_las": MultipleConditionListAccuracyMetric(metric_name="parent_las", dataset_name=tag),  # 
            "previous_uas": MaskedListAccuracyMetric(metric_name="previous_uas", dataset_name=tag),  # previous node UAS, added
            "previous_las": MultipleConditionListAccuracyMetric(metric_name="previous_las", dataset_name=tag),  # previous node acc, both previous id and previous relation should be correct
            "previous_exact_acc": MaskedListAccuracyMetric(metric_name="previous_exact_acc", dataset_name=tag),  # if previous_id == i and previous_relation == "Continue" --> i; if previous_relation == "Break" --> -1
            # "parent_acc_wo_figure": ListAccuracyMetric(metric_name="parent_acc_wo_figure", dataset_name=tag),
            "parent_acc_wo_figure": MaskedListAccuracyMetric(metric_name="parent_acc_wo_figure", dataset_name=tag),
            "parent_acc_attributed": MultipleClassificationAndListAccuracyMetric(metric_name="parent_acc_attributed", dataset_name=tag),  # parent node LAS
            "discourse_acc": SubdiscoursePRFMetric(metric_name="discourse_acc", dataset_name=tag), }

    num_steps = 0

    # max_train_acc, max_dev_acc, max_test_acc = 0, 0, 0
    # max_train_epoch, max_dev_epoch, max_test_epoch = 0, 0, 0
    t0 = time.time()
    for epoch in range(1, int(args.num_train_epochs) + 1):
        time_lag = get_time_lag(t0)
        print(time_lag)
        # t1 = time.time()
        # print(f"{(t1 - t0)//3600} hours, {((t1 - t0) - ((t1 - t0)//3600)*3600)//60} mins, {(t1 - t0) % 60} seconds")
        print("=" * 10 + "Epoch {} / {}".format(epoch, int(args.num_train_epochs)) + "=" * 10)
        model.zero_grad()
        epoch_loss = 0.0
        loss_dict = defaultdict(float)
        loss_dict["loss"] = epoch_loss
        # epoch_father_loss = 0.0
        # epoch_previous_loss = 0.0
        for step, batch in enumerate(tqdm(train_dataloader)):
            # print("number of nodes: %d" % len(batch["input_ids"][0]))
            model.train()
            # with open("ddddd.txt", "a") as f:
            #     f.write(f"batch = {step}"+ "\n")
            #     f.write(str(torch.cuda.memory_allocated())+ "\n")
            # print(batch)
            batch_meta = batch.pop("meta")
            print(batch_meta["ids"])
            # with open("aaa.txt", "a") as a:
            #     a.write(batch_meta["ids"][0])
            # print(batch)
            # print(torch.cuda.memory_allocated(args.device))
            inputs = {k: v.to(args.device) if type(v) == torch.Tensor else v for k, v in batch.items()}
            # print("before forward:", torch.cuda.memory_allocated(args.device))
            # print(inputs.keys())
            outputs = model(**inputs)
            # print("after forward:", torch.cuda.memory_allocated(args.device))
            loss = outputs["loss"]
            # epoch_loss += loss.detach() / args.train_batch_size
            loss_dict.update({
                loss_name: loss_dict[loss_name] + loss_value.detach() / args.train_batch_size
                for loss_name, loss_value in outputs.items()
                if loss_name.endswith("loss") and loss_value is not None
            })
            # epoch_father_loss += outputs[1].detach() / args.train_batch_size
            # epoch_previous_loss += outputs[2].detach() / args.train_batch_size
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            # print("after backward:", torch.cuda.memory_allocated(args.device))
            # scaler.scale(loss).backward()
            if step % args.gradient_accumulation_steps == 0:
                num_steps += 1
                if args.max_grad_norm > 0:
                    # scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # scaler.step(optimizer)
                # scaler.update()
                optimizer.step()
                scheduler.step()
                model.zero_grad()

        print("\t".join([f"{loss_name} = {loss_value}" for loss_name, loss_value in loss_dict.items()]))
        with open(args.log_dir, "a+", encoding="utf-8") as metric_file:
            metric_file.write(f"Epoch = {epoch}" + "\n")
            metric_file.write(get_localtime() + "||" + "Cumulative time:" + get_time_lag(t0) + "\n")
            metric_file.write(
                "\t".join([f"{loss_name} = {loss_value}" for loss_name, loss_value in loss_dict.items()]) + "\n"
            )

        # evaluate per epoch
        for tag, features in benchmarks:
            print("=" * 10 + "Evaluation on {} set".format(tag) + "=" * 10)
            # metrics = evaluate(args, epoch, model, features, tag=tag, evaluation_metrics=evaluation_metrics[tag])
            evaluate(args, epoch, model, features, eval_collate_fn, tag=tag, evaluation_metrics=evaluation_metrics[tag])
            # parent_acc = metrics["parent_acc"]
            with open(args.log_dir, "a+", encoding="utf-8") as metric_file:
                metric_file.write(f"Epoch = {epoch}" + "\n")
                metric_file.write(f"Evaluate on {tag} set" + "\n")
                metric_file.write("\t".join(
                    [f"{metric_name} = {metric.value}" for metric_name, metric in
                     evaluation_metrics[tag].items()]) + "\n")
                # metric_file.write(f"parent_acc = {parent_acc}, previous_acc = {previous_acc}, parent_acc_wo_figure = {parent_acc_wo_figure}" + "\n")

                if evaluation_metrics[tag]["parent_uas"].current_best_epoch == epoch:
                    metric_file.write(f"max parent_uas score on {tag} set update at Epoch {epoch}" + "\n")

                if args.earlystop > 0:
                    if epoch - evaluation_metrics["dev"]["parent_uas"].current_best_epoch >= args.earlystop:
                        print(
                            f"max parent_uas score on dev set do not increase for {args.earlystop} epochs, early stop")
                        metric_file.write(
                            f"max parent_uas score on dev set do not increase for {args.earlystop} epochs, early stop")
                        exit(0)

        # save model checkpoint
        if args.save_checkpoint:
            # torch.save(model.state_dict(), os.path.join(args.model_path, f"checkpoint_{epoch}.pkl"))
            torch.save(model.state_dict(), os.path.join(args.model_checkpoint_dir, f"epoch_{epoch}.pkl"))
            
            print("Model checkpoint saved")


def evaluate(args, epoch, model, features, collate_fn, tag='dev', evaluation_metrics=None):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, collate_fn=collate_fn, drop_last=False)

    model_implemented_funtions = {
        "parent_ids": False,
        "parent_relations": False,
        "previous_ids": False,
        "previous_relations": False,
    }
    list_of_documents = []

    for i_b, batch in enumerate(tqdm(dataloader)):
        model.eval()
        batch_meta = batch.pop("meta")

        # forward pass a batch
        inputs = {k: v.to(args.device) if type(v) == torch.Tensor else v for k, v in batch.items()}
        """if not args.golden_parent_when_evaluate:
            inputs["golden_parent"] = None
            inputs["golden_previous"] = None"""
        # inputs["training_tuple_ids"] = None
        # inputs["golden_parent"] = None  # ???
        # inputs["golden_previous"] = None
        with torch.no_grad():
            # outputs = model(**inputs)
            outputs = model.predict(**inputs)
            if tag=='dev' and i_b == 0:
                # print("outputs_loss", outputs["loss"])
                print("outputs_father_ids", outputs["father_ids"])
                print("outputs_previous_labels", outputs["previous_labels"])
            if i_b == 0:
                model_implemented_funtions.update({
                    "parent_ids": outputs["father_ids"] is not None,
                    "parent_relations": outputs["father_labels"] is not None,
                    "previous_ids": outputs["previous_ids"] is not None,
                    "previous_relations": outputs["previous_labels"] is not None,
                })
            # predict_parent_ids = outputs["father_ids"].reshape(-1).tolist() if type(outputs["father_ids"]) == torch.tensor else outputs["father_ids"]
            # predict_parent_labels = outputs["father_labels"].reshape(-1).tolist() if type(outputs["father_labels"]) == torch.tensor else outputs["father_labels"]
            # predict_previous_ids = outputs["previous_ids"].reshape(-1).tolist() if type(outputs["previous_ids"]) == torch.tensor else outputs["previous_ids"]
            # predict_previous_relations = outputs["previous_labels"].reshape(-1).tolist() if type(outputs["previous_labels"]) == torch.tensor else outputs["previous_labels"]

        # deal with batch meta information
        list_of_documents += DocumentRecorder.from_data(batch_meta, outputs, model_implemented_funtions)

        """preds += [(np.array(predict_parents, dtype=np.int64),
                   np.array(predict_previous, dtype=np.int64))]"""
        # parent_preds += predict_parents
        # previous_preds += predict_previous
        # batch_pred_parent_ids += predict_parent_ids
        # batch_pred_previous_ids += predict_previous_ids
        # batch_pred_parent_relations += predict_parent_labels
        # # batch_pred_previous_relations += predict_previous_relations
        # batch_pred_previous_relations += predict_previous_relations

        # write down logs for each document
        # if i_b < 1:
        #     print("predict parents: ", predict_parents)
        #     print("golden parents: ", parent_golden)
        #     print("predict previous: ", predict_previous)
        #     print("golden previous: ", previous_golden)

    # for case study
    with open(args.log_dir + "_evaluation_output", "a+", encoding="utf-8") as output_file:
        # output_file.write(f"Epoch = {epoch}" + "\n")
        output_file.write("=" * 20 + f"Epoch = {epoch}" + "=" * 20 + "\n")
        """output_file.write(f"Evaluate on {tag} set" + "\t" + f"id = {ids}" + "\n")
        output_file.write(f"golden parent ids: {golden_parent_ids}" + "\n")
        output_file.write(f"predict parent ids: {predict_parent_ids}" + "\n")
        output_file.write(f"{difference_between_list(golden_parent_ids, predict_parent_ids)}" + "\n")"""
        for document_record in list_of_documents:
            output_file.write(f"Evaluate on {tag} set" + "\t")
            output_file.write(str(document_record) + "\n")
        """for i, document_id in enumerate(ids):
            output_file.write(f"Evaluate on {tag} set" + "\t" + f"id = {document_id}" + "\n")
            case_golden_parent_ids = batch_meta['golden_parent_ids'][i]
            case_predict_parent_ids = outputs['father_ids'][i].tolist()
            case_golden_parent_relations = batch_meta['golden_parent_relations'][i]
            case_predict_parent_relations = outputs['father_labels'][i].tolist()
            case_golden_previous_ids = batch_meta['golden_previous_ids'][i]
            case_predict_previous_ids = outputs['previous_ids'][i].tolist()
            case_golden_previous_relations = batch_meta['golden_previous_relations'][i]
            case_predict_previous_relations = outputs['previous_labels'][i].tolist()
            output_file.write(f"golden parent ids: {case_golden_parent_ids}" + "\n")
            output_file.write(f"predict parent ids: {case_predict_parent_ids}" + "\n")
            output_file.write(f"{difference_between_list(case_golden_parent_ids, case_predict_parent_ids)}" + "\n")
            output_file.write(f"golden parent relations: {case_golden_parent_relations}" + "\n")
            output_file.write(f"predict parent relations: {case_predict_parent_relations}" + "\n")
            output_file.write(f"golden previous ids: {case_golden_previous_ids}" + "\n")
            output_file.write(f"predict previous ids: {case_predict_previous_ids}" + "\n")
            output_file.write(f"golden previous relations: {case_golden_previous_relations}" + "\n")
            output_file.write(f"predict previous relations: {case_predict_previous_relations}" + "\n")"""
        # output_file.write(f"golden previous ids: {[i - 1 for i in golden_previous_ids][1:]}" + "\n")
        # output_file.write(f"golden previous: {golden_previous_relations}" + "\n")
        # output_file.write(f"predict previous: {predict_previous_relations}" + "\n")
        # output_file.write(f"{difference_between_list(golden_previous_relations, predict_previous_relations)}" + "\n")

    """evaluation_inputs = {
        
    }
    for metric in evaluation_metrics.values():
        metric.update(**evaluation_inputs)"""
    # evaluation metrics
    """self.golden_parent_ids = golden_parent_ids or []
        self.golden_parent_relations = golden_parent_relations or []
        self.golden_previous_ids = golden_previous_ids or []
        self.golden_previous_relations = golden_previous_relations or []
        self.predicted_parent_ids = predicted_parent_ids or []
        self.predicted_parent_relations = predicted_parent_relations or []
        self.predicted_previous_ids = predicted_previous_ids or []
        self.predicted_previous_relations = predicted_previous_relations or []
        self.id = id
        self.node_modal = node_modal or []"""
    total_golden_parent_ids = flatten_list([document_record.golden_parent_ids for document_record in list_of_documents])
    total_pred_parent_ids = flatten_list([document_record.predicted_parent_ids for document_record in list_of_documents])
    total_golden_parent_relations = flatten_list([document_record.golden_parent_relations for document_record in list_of_documents])
    total_pred_parent_relations = flatten_list([document_record.predicted_parent_relations for document_record in list_of_documents])
    total_golden_previous_ids = flatten_list([document_record.golden_previous_ids for document_record in list_of_documents])
    total_pred_previous_ids = flatten_list([document_record.predicted_previous_ids for document_record in list_of_documents])
    total_golden_previous_relations = flatten_list([document_record.golden_previous_relations for document_record in list_of_documents])
    total_pred_previous_relations = flatten_list([document_record.predicted_previous_relations for document_record in list_of_documents])
    total_node_modals = flatten_list([document_record.node_modal for document_record in list_of_documents])
    length_of_documents = [len(document_record.golden_parent_ids) for document_record in list_of_documents]

    # if tag=='dev':
        # print("total_golden_parent_ids", total_golden_parent_ids)
        # print("total_pred_parent_ids", total_pred_parent_ids)
        # print("total_golden_previous_ids", total_golden_previous_ids)
        # print("total_pred_previous_ids", total_pred_previous_ids)
        # print("total_golden_previous_relations", total_golden_previous_relations)
        # print("total_pred_previous_relations", total_pred_previous_relations)

    if model_implemented_funtions["parent_ids"]:
        # UAS, unlabeled attachment score
        evaluation_metrics["parent_uas"].update(
            golden=total_golden_parent_ids,
            pred=total_pred_parent_ids,
        )

        # UAS, unlabeled attachment score, without considering figures
        evaluation_metrics["parent_acc_wo_figure"].update(
            golden=total_golden_parent_ids,
            pred=total_pred_parent_ids,
            mask=[1 if (total_node_modals[i] != "Figure" and total_node_modals[i] != "Figure&Title") else 0
                  for i, id in enumerate(total_node_modals)],
        )

        if model_implemented_funtions["parent_relations"]:
            evaluation_metrics["parent_las"].update( 
                goldens=(total_golden_parent_ids, total_golden_parent_relations),
                preds=(total_pred_parent_ids, total_pred_parent_relations),
            )
            # LAS, labeled attachment score
            evaluation_metrics["parent_acc_attributed"].update(
                parent_goldens=total_golden_parent_ids,
                parent_preds=total_pred_parent_ids,
                parent_relation_goldens=total_golden_parent_relations,
                parent_relation_preds=total_pred_parent_relations,
                id2classmapper=FATHER_RELATION_dict,
            )

    if model_implemented_funtions["previous_ids"] and model_implemented_funtions["previous_relations"]:
        # segmentation acc
        """evaluation_metrics["previous_acc"].update(
            goldens=total_golden_previous_relations,
            preds=total_pred_previous_relations,
            mask_id=3)"""
        evaluation_metrics["previous_uas"].update( 
            golden=total_golden_previous_ids,
            pred=total_pred_previous_ids,
        )
        evaluation_metrics["previous_las"].update(
            goldens=(total_golden_previous_ids, total_golden_previous_relations),
            preds=(total_pred_previous_ids, total_pred_previous_relations),
        )
        evaluation_metrics["previous_exact_acc"].update(
            golden=[node_id if label != "Break" else -1 for node_id, label in zip(total_golden_previous_ids, total_golden_previous_relations)],
            pred=[node_id if label != "Break" else -1 for node_id, label in zip(total_pred_previous_ids, total_pred_previous_relations)],
        )

    if model_implemented_funtions["parent_ids"] and model_implemented_funtions["previous_ids"] and model_implemented_funtions["previous_relations"]:
        evaluation_metrics["discourse_acc"].update(
            parent_goldens=total_golden_parent_ids,
            parent_preds=total_pred_parent_ids,
            previous_goldens=total_golden_previous_relations,
            previous_preds=total_pred_previous_relations,
            length_of_discourse=length_of_documents)

    metrics = {metric_name: evaluation_metrics[metric_name].value for metric_name in evaluation_metrics.keys()}
    """# keys = np.array(keys, dtype=np.int64)
    # preds = np.array(preds, dtype=np.int64)
    # P, R, max_f1, metrics_by_type = get_f1(keys, preds,)
    parent_acc = get_acc(parent_goldens, parent_preds, mask_id=1000)
    previous_acc = get_acc(previous_goldens, previous_preds, mask_id=3)
    metrics = {"parent_acc": parent_acc, "previous_acc": previous_acc, }

    parent_acc_wo_figure = get_acc(
        [id if (node_modals[i] != "Figure" and node_modals[i] != "Figure&Title") else 1000 for i, id in
         enumerate(parent_goldens)], parent_preds, mask_id=1000)
    metrics.update({
                       "parent_acc_wo_figure": parent_acc_wo_figure})  # TODO Do not consider whether the father node of a figure is correctly predicted
    # parent_acc_correct_figure = get_acc([id if (node_modals[i]!="Figure" and node_modals[i]!="Figure&Title") else 1000 for i, id in enumerate(parent_goldens)], parent_preds, mask_id=1000)
    # metrics.update({"parent_acc_correct_figure": parent_acc_correct_figure})"""

    # print(f"parent_acc = {parent_acc}, previous_acc = {previous_acc}")
    print(", ".join([f"{metric_name} = {metric_value}" for metric_name, metric_value in metrics.items()]))

    # output = {tag + "_precision": P * 100, tag + "_recall": R * 100, tag + "_f1": max_f1 * 100, }
    # print(output)
    # print(metrics_by_type)
    # return max_f1, output
    # return parent_acc, previous_acc
    return metrics


# def main(model_type):
def main():

    parser = prepare_all_argparsers(train_env_dict)
    args = parser.parse_args()

    model_type = args.model_name
    print(f"model type == {model_type}")
    train_env = train_env_dict[model_type]


    """add_common_arguments(parser)
    args = parser.parse_args()
    args.model_type = model_type"""
    args.log_dir = f"{model_type}_{get_localtime()}_{args.log_dir}"
    if not os.path.exists("log"):
        os.mkdir("log")
    args.log_dir = os.path.join("log", args.log_dir)
    
    args.model_checkpoint_dir = os.path.join(args.model_path, f"{model_type}_{get_localtime()}")
    
    args.n_gpu = torch.cuda.device_count()
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    print(f"using device {device}")
    args.device = device
    if args.seed > 0:
        set_seed(args)
    
    args.parent_relation_dims = len(list(FATHER_RELATION_dict.keys())) - 1 if not args.use_parent_relation_fine else len(list(FATHER_RELATION_FINE_dict.keys())) - 1
    args.parent_relation_mapping = FATHER_RELATION_dict if not args.use_parent_relation_fine else FATHER_RELATION_FINE_dict
    # args.previous_relation_dims = 2 if args.combine_before else 3
    args.previous_relation_dims = len(list(PREVIOUS_RELATION_dict.keys())) - 1 if not args.use_previous_relation_fine else len(list(PREVIOUS_RELATION_dict.keys())) - 1
    args.previous_relation_mapping = PREVIOUS_RELATION_dict.copy() if not args.use_parent_relation_fine else PREVIOUS_RELATION_dict
    # if args.combine_before:
    #     args.previous_relation_mapping.pop(2)

    # _, _, labelfield = build_vocab_from_dataset(train_file)
    # labelfield.add_one("None")

    """tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )"""
    # processor = WebDataProcessor(args, tokenizer)
    tokenizer = train_env.prepare_tokenizer(args)
    # processor = processor_dict[model_type](args, tokenizer)
    processor = train_env.prepare_dataprocessor(args, tokenizer)
    # args.data_cache_key += processor.get_data_process_version_tag(args)
    if os.path.exists(os.path.join(args.data_cache_path, f"{model_type}_{args.data_cache_key}_train.pkl")) and args.load_cached_features:
        print("Loading features from archive ...")
        train_feature_file = open(os.path.join(args.data_cache_path, f"{model_type}_{args.data_cache_key}_train.pkl"), 'rb')
        train_features = pickle.load(train_feature_file)
        train_feature_file.close()
        dev_feature_file = open(os.path.join(args.data_cache_path, f"{model_type}_{args.data_cache_key}_dev.pkl"), 'rb')
        dev_features = pickle.load(dev_feature_file)
        dev_feature_file.close()
        test_feature_file = open(os.path.join(args.data_cache_path, f"{model_type}_{args.data_cache_key}_test.pkl"), 'rb')
        test_features = pickle.load(test_feature_file)
        test_feature_file.close()
        
        html_tag_vocab = Vocab.from_token2id_mapper_file(args.html_vocab_dir)
        xpath_tag_vocab = Vocab.from_token2id_mapper_file(args.xpath_vocab_dir)
        processor.set_html_vocab(html_tag_vocab)
        processor.set_xpath_vocab(xpath_tag_vocab)

    else:
        print("Building features from scratch ...")
        train_ds = WebDataset(args.train_set_dir)
        dev_ds = WebDataset(args.dev_set_dir)
        test_ds = WebDataset(args.test_set_dir)
        train_features = processor.get_features_from_dataset(train_ds)
        dev_features = processor.get_features_from_dataset(dev_ds)
        test_features = processor.get_features_from_dataset(test_ds)
        
        html_tag_vocab = processor.get_html_vocab()
        html_tag_vocab.save_vocab_to_token2id_mapper(args.html_vocab_dir)
        xpath_tag_vocab = processor.get_xpath_vocab()
        xpath_tag_vocab.save_vocab_to_token2id_mapper(args.xpath_vocab_dir)
        
        features_dict = {"train": train_features, "dev": dev_features, "test": test_features}
        for name in features_dict.keys():
            saved_feature_file = open(os.path.join(args.data_cache_path, f"{model_type}_{args.data_cache_key}_{name}.pkl"), "wb")
            pickle.dump(features_dict[name], saved_feature_file)
            saved_feature_file.close()

    # print(train_features[:5])
    print("number of train instances : %d" % len(train_features))
    print("instance keys :", train_features[0].keys())
    print("instance[0] :", train_features[0])

    """config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        # num_labels=args.num_class,
    )
    config.gradient_checkpointing = True  # TODO"""
    # if args.test_only:
    #     config = AutoConfig.from_pretrained(
    #         args.config_name if args.config_name else args.model_name_or_path,
    #         # num_labels=args.num_class,
    #     )
    #     config.gradient_checkpointing = True  # TODO
    #
    #     model = model_dict[model_type](args, config)
    #     load_path = os.path.join(args.model_path, f"checkpoint_{args.test_checkpoint_id}.pkl")
    #     print(f"Loading NN model pretrained checkpoint from {load_path} ...")
    #     model.load_state_dict(torch.load(load_path))
    #     model.to(args.device)
    #     evaluate(args, model, test_features, tag='test')
    #     exit(0)

    """if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
    else:
        print(f"args.model_path: {args.model_path} already exists, still go on? y/[n]")
        c = input()
        if c != "y":
            exit(10)"""
    if not os.path.exists(args.model_checkpoint_dir):
        os.makedirs(args.model_checkpoint_dir)
    else:
        print(f"args.model_checkpoint_dir: {args.model_checkpoint_dir} already exists, still go on? y/[n]")
        c = input()
        if c != "y":
            exit(10)

    # model = BaselineModel(args, config)
    # model = model_dict[model_type](args, config)
    # model.to(args.device)
    model = train_env.prepare_model(args, tokenizer, processor)

    print("model prepared")

    if args.test_only:
        # load_path = os.path.join(args.model_path, f"checkpoint_{args.test_checkpoint_id}.pkl")
        # print(f"Loading NN model pretrained checkpoint from {load_path} ...")
        # model.load_state_dict(torch.load(load_path))
        # model.to(args.device)
        evaluate(args, 0, model, test_features, tag='test', collate_fn=train_env.get_test_collate_fn(processor))
        exit(0)

    # if len(processor.new_tokens) > 0:
    #     model.encoder.resize_token_embeddings(len(tokenizer))  # TODO

    benchmarks = (
        ("train", train_features),
        ("dev", dev_features),
        ("test", test_features),
    )

    train(args, train_env, model, train_features, benchmarks, train_env.get_train_collate_fn(processor), train_env.get_test_collate_fn(processor))


if __name__ == '__main__':
    main()
    
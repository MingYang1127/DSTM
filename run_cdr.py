import argparse
import glob
import json
import logging
import os
import pickle
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup
)

from model import (BertForDocRED, RobertaForDocRED)
from loader import CDRDataLoader
from cdr_dataset import CDR_Dataset

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    if args.warmup_ratio > 0:
        args.warmup_steps = t_total * args.warmup_ratio

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]

    # set grad: train all params
    for name, param in model.named_parameters():
        param.requires_grad = True

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.lr_schedule == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to global_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            # roberta does not accept token_type_ids
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2] if args.model_type == 'bert' else None,
                      "ent_mask": batch[3],
                      "ent_ner": batch[4],
                      "ent_pos": batch[5],
                      "ent_distance": batch[6],
                      "structure_mask": batch[7],
                      "label": batch[8],
                      "label_mask": batch[9],
                      "adj": batch[10],
                      "gauss_p": batch[11],
                      }

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                len(epoch_iterator) <= args.gradient_accumulation_steps
                and (step + 1) == len(epoch_iterator)
            ):
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    print(json.dumps({**logs, **{"step": global_step}}))

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def evaluate(args, loader, eval_dataset, model, tokenizer, prefix=""):

    # processor = DocREDProcessor()
    # dev_examples = processor.get_dev_examples(args.data_dir)

    # label_map = processor.get_label_map(args.data_dir)
    # predicate_map = {}
    # for predicate in label_map.keys():
    #     predicate_map[label_map[predicate]] = predicate
    # loader = CDRDataLoader(os.path.join(args.data_dir, args.eval_file))
    # loader()
    doc_id_all = loader.documents.keys()
    # eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True, predict=False)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    ent_masks = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2] if args.model_type == 'bert' else None,
                      "ent_mask": batch[3],
                      "ent_ner": batch[4],
                      "ent_pos": batch[5],
                      "ent_distance": batch[6],
                      "structure_mask": batch[7],
                      "label": batch[8],
                      "label_mask": batch[9],
                      "adj": batch[10],
                      "gauss_p": batch[11],
                      }

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            label_masks = inputs["label_mask"].detach().cpu().numpy()
            # labels = inputs["label"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            label_masks = np.append(label_masks, inputs["label_mask"].detach().cpu().numpy(), axis=0)
            # labels = np.append(labels, inputs["label"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    tp_cross, tn_cross, fp_cross, fn_cross = 0, 0, 0, 0
    tp_noncross, tn_noncross, fp_noncross, fn_noncross = 0, 0, 0, 0
    truncated_pairs_count = 0
    total_pairs = 0
    total_pairs_crossed = 0
    eval_result_correct = []
    eval_result_wrong = []
    for (i, (doc_id, pred, label_mask)) in enumerate(zip(doc_id_all, preds, label_masks)):
        entities = loader.entities[doc_id]
        labels = loader.pairs[doc_id]
        entities_list = [ent_id for ent_id in entities.keys()]

        # for calculate the CDR
        for pair in labels.keys():
            eval_correct = {}
            eval_wrong = {}
            if labels[pair].cross == 'CROSS':
                total_pairs_crossed += 1
            total_pairs += 1
            s_id = entities_list.index(pair[0])
            o_id = entities_list.index(pair[1])
            if labels[pair].direction == 'L2R':
                ent_h = s_id
                ent_t = o_id
            elif labels[pair].direction == 'R2L':
                ent_h = o_id
                ent_t = s_id
            # ent_h = s_id
            # ent_t = o_id
            logit_s_o = pred[ent_h][ent_t]
            y = logit_s_o.argmax(-1)
            # ent was truncated, give NR as pred
            # if label_mask[ent_h][ent_t] == 0 or (label_mask[ent_h][ent_t] == 1 and y == 0):
            if (label_mask[ent_h][ent_t] == 1 and y == 0):
                if labels[pair].type == '1:NR:2':
                    if labels[pair].cross == 'CROSS':
                        tn_cross += 1
                    else:
                        tn_noncross += 1
                    eval_correct[doc_id] = [pair, labels[pair].type]
                elif labels[pair].type == '1:CID:2':
                    if labels[pair].cross == 'CROSS':
                        fn_cross += 1
                    else:
                        fn_noncross += 1
                    eval_wrong[doc_id] = [pair, labels[pair].type]
            # elif (label_mask[ent_h][ent_t] == 1 and y[ent_h][ent_t] == 1):
            elif (label_mask[ent_h][ent_t] == 1 and y == 1):
                if labels[pair].type == '1:NR:2':
                    if labels[pair].cross == 'CROSS':
                        fp_cross += 1
                    else:
                        fp_noncross += 1
                    eval_wrong[doc_id] = [pair, labels[pair].type]
                elif labels[pair].type == '1:CID:2':
                    if labels[pair].cross == 'CROSS':
                        tp_cross += 1
                    else:
                        tp_noncross += 1
                    eval_correct[doc_id] = [pair, labels[pair].type]
            else:
                continue
                # sys.exit('No prediction')
            if eval_correct != {}:
                eval_result_correct.append(eval_correct)
            if eval_wrong != {}:
                eval_result_wrong.append(eval_wrong)

    tp, fp, tn, fn = tp_cross + tp_noncross, fp_cross + fp_noncross, tn_cross + tn_noncross, fn_cross + fn_noncross
    precision, recall = tp / (tp + fp or 1), tp / (tp + fn or 1)
    f1 = 2 * precision * recall / (precision + recall or 1)

    precision_cross, recall_cross = tp_cross / (tp_cross + fp_cross or 1), tp_cross / (tp_cross + fn_cross or 1)
    f1_cross = 2 * precision_cross * recall_cross / (precision_cross + recall_cross or 1)

    precision_noncross, recall_noncross = tp_noncross / (tp_noncross + fp_noncross or 1), tp_noncross / (
            tp_noncross + fn_noncross or 1)
    f1_noncross = 2 * precision_noncross * recall_noncross / (precision_noncross + recall_noncross or 1)

    result = {"loss": eval_loss, "precision": round(precision, 4), "recall": round(recall, 4), "micro-f1": round(f1, 4),
              "precision_cross": round(precision_cross, 4), "recall_cross": round(recall_cross, 4),
              "micro-f1_cross": round(f1_cross, 4),
              "precision_noncross": round(precision_noncross, 4), "recall_noncross": round(recall_noncross, 4),
              "micro-f1_noncross": round(f1_noncross, 4),
              "total pairs": total_pairs, "truncated pairs count": truncated_pairs_count,
              "crossed pairs": total_pairs_crossed}
    # cmd = "echo -e {} >> ./output/results.txt".format(result)
    # os.system(cmd)
    # write pred file
    output_eval_file_0 = os.path.join(args.checkpoint_dir, "eval_result_correct.json")
    with open(output_eval_file_0, 'w') as f:
        json.dump(eval_result_correct, f)
    output_eval_file_1 = os.path.join(args.checkpoint_dir, "eval_result_wrong.json")
    with open(output_eval_file_1, 'w') as f:
        json.dump(eval_result_wrong, f)

    print(result)

    return result


def predict(args, loader, test_dataset, model, tokenizer, prefix=""):

    doc_id_all = loader.documents.keys()
    # test_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, predict=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    ent_masks = None
    out_label_ids = None
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2] if args.model_type == 'bert' else None,
                      "ent_mask": batch[3],
                      "ent_ner": batch[4],
                      "ent_pos": batch[5],
                      "ent_distance": batch[6],
                      "structure_mask": batch[7],
                      "label": batch[8],
                      "label_mask": batch[9],
                      "adj": batch[10],
                      "gauss_p": batch[11],
                      }

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            label_masks = inputs["label_mask"].detach().cpu().numpy()
            labels = inputs["label"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            label_masks = np.append(label_masks, inputs["label_mask"].detach().cpu().numpy(), axis=0)
            labels = np.append(labels, inputs["label"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    tp_cross, tn_cross, fp_cross, fn_cross = 0, 0, 0, 0
    tp_noncross, tn_noncross, fp_noncross, fn_noncross = 0, 0, 0, 0
    truncated_pairs_count = 0
    total_pairs = 0
    total_pairs_crossed = 0
    result_correct = []
    result_wrong = []
    for (i, (doc_id, pred, label_mask, label)) in enumerate(zip(doc_id_all, preds, label_masks, labels)):
        entities = loader.entities[doc_id]
        labels = loader.pairs[doc_id]
        entities_list = [ent_id for ent_id in entities.keys()]

        # for calculate the CDR
        for pair in labels.keys():
            correct = {}
            wrong = {}
            if labels[pair].cross == 'CROSS':
                total_pairs_crossed += 1
            total_pairs += 1
            s_id = entities_list.index(pair[0])
            o_id = entities_list.index(pair[1])
            if labels[pair].direction == 'L2R':
                ent_h = s_id
                ent_t = o_id
            elif labels[pair].direction == 'R2L':
                ent_h = o_id
                ent_t = s_id
            # ent_h = s_id
            # ent_t = o_id
            logit_s_o = pred[ent_h][ent_t]
            y = logit_s_o.argmax(-1)
            # ent was truncated, give NR as pred
            # if label_mask[ent_h][ent_t] == 0 or (label_mask[ent_h][ent_t] == 1 and y == 0):
            if (label_mask[ent_h][ent_t] == 1 and y == 0):
                # if labels[pair].type == '1:NR:2' or labels[pair].type == 'not_include':
                if labels[pair].type == '1:NR:2':
                    if labels[pair].cross == 'CROSS':
                        tn_cross += 1
                    else:
                        tn_noncross += 1
                    correct[doc_id] = [pair, labels[pair].type]
                elif labels[pair].type == '1:CID:2':
                    if labels[pair].cross == 'CROSS':
                        fn_cross += 1
                    else:
                        fn_noncross += 1
                    wrong[doc_id] = [pair, labels[pair].type]
            # elif (label_mask[ent_h][ent_t] == 1 and y[ent_h][ent_t] == 1):
            elif (label_mask[ent_h][ent_t] == 1 and y == 1):
                # if labels[pair].type == '1:NR:2' or labels[pair].type == 'not_include':
                if labels[pair].type == '1:NR:2':
                    if labels[pair].cross == 'CROSS':
                        fp_cross += 1
                    else:
                        fp_noncross += 1
                    wrong[doc_id] = [pair, labels[pair].type]
                elif labels[pair].type == '1:CID:2':
                    if labels[pair].cross == 'CROSS':
                        tp_cross += 1
                    else:
                        tp_noncross += 1
                    correct[doc_id] = [pair, labels[pair].type]
            else:
                continue
                # sys.exit('No prediction')
            if correct != {}:
                result_correct.append(correct)
            if wrong != {}:
                result_wrong.append(wrong)

    tp, fp, tn, fn = tp_cross + tp_noncross, fp_cross + fp_noncross, tn_cross + tn_noncross, fn_cross + fn_noncross
    precision, recall = tp / (tp + fp or 1), tp / (tp + fn or 1)
    f1 = 2 * precision * recall / (precision + recall or 1)

    precision_cross, recall_cross = tp_cross / (tp_cross + fp_cross or 1), tp_cross / (tp_cross + fn_cross or 1)
    f1_cross = 2 * precision_cross * recall_cross / (precision_cross + recall_cross or 1)

    precision_noncross, recall_noncross = tp_noncross / (tp_noncross + fp_noncross or 1), tp_noncross / (
            tp_noncross + fn_noncross or 1)
    f1_noncross = 2 * precision_noncross * recall_noncross / (precision_noncross + recall_noncross or 1)

    result = {"loss": eval_loss, "precision": round(precision, 4), "recall": round(recall, 4), "micro-f1": round(f1, 4),
              "precision_cross": round(precision_cross, 4), "recall_cross": round(recall_cross, 4),
              "micro-f1_cross": round(f1_cross, 4),
              "precision_noncross": round(precision_noncross, 4), "recall_noncross": round(recall_noncross, 4),
              "micro-f1_noncross": round(f1_noncross, 4),
              "total pairs": total_pairs, "truncated pairs count": truncated_pairs_count,
              "crossed pairs": total_pairs_crossed}
    output_test_file = os.path.join(args.checkpoint_dir, "result_correct.json")
    with open(output_test_file, 'w') as f:
        json.dump(result_correct, f)
    output_test_file = os.path.join(args.checkpoint_dir, "result_wrong.json")
    with open(output_test_file, 'w') as f:
        json.dump(result_wrong, f)

    print(result)

    return result


def load_and_cache_examples(args, loader, tokenizer, evaluate=False, predict=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data
    logger.info("Creating features from dataset file at %s", args.data_dir)

    if evaluate:
        file_dev_graph = open('D:/code/DSPM_CDR/data/bert_dev.graph', 'rb')
        ex_index2graph = pickle.load(file_dev_graph)
        file_dev_graph.close()
    elif predict:
        file_test_graph = open('D:/code/DSPM_CDR/data/bert_test.graph', 'rb')
        ex_index2graph = pickle.load(file_test_graph)
        file_test_graph.close()
    else:
        file_train_graph = open('D:/code/DSPM_CDR/data/bert_train.graph', 'rb')
        ex_index2graph = pickle.load(file_train_graph)
        file_train_graph.close()

    features = CDR_Dataset(
        loader,
        ex_index2graph,
        args.model_type,
        tokenizer,
        max_length=args.max_seq_length,
        max_ent_cnt=args.max_ent_cnt,
    )

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_ent_cnt",
        default=42,
        type=int,
        help="The maximum entities considered.",
    )
    parser.add_argument("--no_naive_feature", action="store_true",
                        help="do not exploit naive features for DocRED, include ner tag, entity id, and entity pair distance")
    parser.add_argument("--entity_structure", default='biaffine', type=str, choices=['none', 'decomp', 'biaffine'],
                        help="whether and how do we incorporate entity structure in Transformer models.")
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default='./checkpoints',
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run pred on the pred set.")
    parser.add_argument("--predict_thresh", default=0.5, type=float, help="pred thresh")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=30, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_ratio", default=0, type=float, help="Linear warmup ratio, overwriting warmup_steps.")
    parser.add_argument("--lr_schedule", default='linear', type=str, choices=['linear', 'constant'],
                        help="Linear warmup ratio, overwriting warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        required=True,
        help="The input data file for training.",
    )
    parser.add_argument(
        "--eval_file",
        default=None,
        type=str,
        required=True,
        help="The input data file for testing.",
    )
    parser.add_argument(
        "--test_file",
        default=None,
        type=str,
        required=True,
        help="The input data file for testing.",
    )
    parser.add_argument(
        "--num_labels",
        default=2,
        type=int,
        help="num of relation class",
    )
    args = parser.parse_args()

    ModelArch = None
    if args.model_type == 'roberta':
        ModelArch = RobertaForDocRED
    elif args.model_type == 'bert':
        ModelArch = BertForDocRED

    if args.no_naive_feature:
        with_naive_feature = False
    else:
        with_naive_feature = False

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
        output_attentions=True,
    )
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model = ModelArch.from_pretrained(args.model_name_or_path,
                                          from_tf=bool(".ckpt" in args.model_name_or_path),
                                          config=config,
                                          cache_dir=args.cache_dir if args.cache_dir else None,
                                          num_labels=args.num_labels,
                                          max_ent_cnt=args.max_ent_cnt,
                                          with_naive_feature=with_naive_feature,
                                          entity_structure=args.entity_structure,
                                          )
        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
        model.to(args.device)
        logger.info("Training parameters %s", args)
        loader = CDRDataLoader(os.path.join(args.data_dir, args.train_file))
        loader()
        train_dataset = load_and_cache_examples(args, loader, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            # Create output directory if needed
            if args.output_dir is None and args.local_rank in [-1, 0]:
                raise ValueError('checkpoint_dir is not set!')
            if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(args.output_dir)

            logger.info("Saving model checkpoint to %s", args.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if args.do_train:
            args.checkpoint_dir = args.output_dir
        elif args.checkpoint_dir is None and args.local_rank in [-1, 0]:
            raise ValueError('checkpoint_dir is not set!')
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, do_lower_case=args.do_lower_case)
        model = ModelArch.from_pretrained(args.checkpoint_dir,
                                          from_tf=bool(".ckpt" in args.model_name_or_path),
                                          config=config,
                                          cache_dir=args.cache_dir if args.cache_dir else None,
                                          num_labels=args.num_labels,
                                          max_ent_cnt=args.max_ent_cnt,
                                          with_naive_feature=with_naive_feature,
                                          entity_structure=args.entity_structure,
                                          )
        model.to(args.device)
        loader = CDRDataLoader(os.path.join(args.data_dir, args.eval_file))
        loader()
        eval_dataset = load_and_cache_examples(args, loader, tokenizer, evaluate=True, predict=False)
        result = evaluate(args, loader, eval_dataset, model, tokenizer)
        results.update(result)

    # predict
    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, do_lower_case=args.do_lower_case)
        model = ModelArch.from_pretrained(args.checkpoint_dir,
                                          from_tf=bool(".ckpt" in args.model_name_or_path),
                                          config=config,
                                          cache_dir=args.cache_dir if args.cache_dir else None,
                                          num_labels=args.num_labels,
                                          max_ent_cnt=args.max_ent_cnt,
                                          with_naive_feature=with_naive_feature,
                                          entity_structure=args.entity_structure,
                                          )
        model.to(args.device)
        loader = CDRDataLoader(os.path.join(args.data_dir, args.test_file))
        loader()
        test_dataset = load_and_cache_examples(args, loader, tokenizer, evaluate=False, predict=True)
        predict(args, loader, test_dataset, model, tokenizer)

    return results

if __name__ == "__main__":
    main()

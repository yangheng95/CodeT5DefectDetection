# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import os
import logging
import argparse
import math
import random

import autocuda
import numpy as np
from io import open

from metric_visualizer import MetricVisualizer
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)
import multiprocessing
import time

from models import DefectModel
from data_utils import get_filenames, get_elapse_time, load_and_cache_defect_data
from models import get_model_size

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}

cpu_cont = multiprocessing.cpu_count()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(args, model, eval_examples, eval_data, write_to_pred=False):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Num batches = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    c_logits = []
    labels = []
    c_labels = []
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        c_label = batch[2].to(args.device)
        with torch.no_grad():
            lm_loss, logit, c_logit = model(inputs, label, c_label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            c_logits.append(c_logit.cpu().numpy())
            labels.append(label.cpu().numpy())
            c_labels.append(c_label.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    c_logits = np.concatenate(c_logits, 0)
    labels = np.concatenate(labels, 0)
    c_labels = np.concatenate(c_labels, 0)
    preds = logits[:, 1] > 0.5
    c_preds = c_logits[:, 1] > 0.5
    eval_acc = np.mean(labels == preds)
    eval_corrupt_acc = np.mean(c_labels == c_preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    result = {
        "eval_loss": float(perplexity),
        "eval_acc": round(eval_acc, 4)*100,
        "eval_corrupt_acc": round(eval_corrupt_acc, 4)*100,
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    if write_to_pred:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:
            for example, pred in zip(eval_examples, preds):
                if pred:
                    f.write(str(example.idx) + '\t1\n')
                else:
                    f.write(str(example.idx) + '\t0\n')

    return result


def main(mv):
    parser = argparse.ArgumentParser()
    t0 = time.time()
    parser.add_argument("--task", type=str, default='defect')
    parser.add_argument("--sub_task", type=str, default='none')
    parser.add_argument("--lang", type=str, default='c')
    parser.add_argument("--model_type", default="codet5", type=str, choices=['roberta', 'bart', 'codet5'])
    parser.add_argument("--add_lang_ids", action='store_true')
    parser.add_argument("--data_num", default=-1, type=int)
    parser.add_argument("--corrupt_instance", default=5, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_train_epochs", default=20, type=int)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--tokenizer_path", default='../tokenizer/salesforce', type=str)
    parser.add_argument("--data_dir", type=str, default='data')
    parser.add_argument("--res_file_name", type=str, default='{}_{}.txt'.format('output', 'defect', 'codet5_small'))
    parser.add_argument("--add_task_prefix", action='store_true', help="Whether to add task prefix for t5 and codet5")
    parser.add_argument("--save_last_checkpoints", default=True, action='store_true')
    parser.add_argument("--always_save_model", default=True, action='store_true')
    parser.add_argument("--do_eval_bleu", default=True, action='store_true', help="Whether to evaluate bleu on dev set.")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    ## Required parameters
    parser.add_argument("--model_name_or_path", default="Salesforce/codet5-small", type=str,
                        help="Path to pre-trained model: e.g. roberta-small")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")

    parser.add_argument("--config_name", default="Salesforce/codet5-small", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="Salesforce/codet5-small", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_source_length", default=512, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=1, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", default=True, action='store_true',
                        help="Whether to run eval on the train set.")
    parser.add_argument("--do_eval", default=True, action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", default=True, action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--save_steps", default=-1, type=int, )
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=1000, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=random.randint(0, 10000),
                        help="random seed for initialization")
    args = parser.parse_args()

    run_path = '{0}/{1}/{2}_lr{3}_bs{4}_src{5}_tgt{6}_patience{7}_epoch{8}'.format(args.data_dir,
                                                                                   args.task,
                                                                                   os.path.basename(args.model_name_or_path),
                                                                                   args.learning_rate,
                                                                                   args.train_batch_size,
                                                                                   args.max_source_length,
                                                                                   args.max_target_length,
                                                                                   args.patience,
                                                                                   args.num_train_epochs
                                                                                   )
    args.cache_path = os.path.join(run_path, 'cache')
    args.summary_dir = os.path.join(run_path, 'tensorboard')
    args.res_dir = os.path.join(run_path, 'prediction')
    args.output_dir = os.path.join(run_path, 'checkpoint')
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda:{}".format(autocuda.auto_cuda_index()) if torch.cuda.is_available() and not args.no_cuda else "cpu")
        # device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        # args.n_gpu = torch.cuda.device_count()
        args.n_gpu = 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, cpu count: %d",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), cpu_cont)
    args.device = device

    set_seed(args)

    # # Build model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.corrupt_instance = args.corrupt_instance

    model = model_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)

    model = DefectModel(model, config, tokenizer, args)
    logger.info("Finish loading model [%s] from %s", get_model_size(model), args.model_name_or_path)

    model.to(device)

    if args.load_model_path is not None:
        logger.info("Reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    pool = multiprocessing.Pool(cpu_cont)
    # pool = None
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    fa = open(args.output_dir + '/summary.log', 'w')

    if args.local_rank in [-1, 0] and args.data_num == -1:
        summary_fn = args.summary_dir
        tb_writer = SummaryWriter(summary_fn)

    if args.do_train:

        # Prepare training data loader
        train_examples, train_data = load_and_cache_defect_data(args, args.train_filename, pool, tokenizer, 'train', is_sample=False)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data, num_replicas=4)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)

        save_steps = max(len(train_dataloader), 1)

        # save_steps = max(len(train_dataloader), 1) // 3
        # args.patience = args.patience * 3

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        if args.warmup_steps < 1:
            warmup_steps = num_train_optimization_steps * args.warmup_steps
        else:
            warmup_steps = int(args.warmup_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)

        global_step, best_acc = 0, 0
        not_acc_inc_cnt = 0
        is_early_stop = False
        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            model.train()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(device) for t in batch)
                source_ids, labels, c_labels = batch
                loss, logits, c_logits = model(source_ids, labels, c_labels)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / nb_tr_steps, 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

                if (step + 1) % save_steps == 0 and args.do_eval:
                    logger.info("***** CUDA.empty_cache() *****")
                    torch.cuda.empty_cache()

                    eval_examples, eval_data = load_and_cache_defect_data(args, args.dev_filename, pool, tokenizer, 'valid', is_sample=False)

                    result = evaluate(args, model, eval_examples, eval_data)
                    eval_acc = result['eval_acc']

                    if args.data_num == -1:
                        tb_writer.add_scalar('dev_acc', round(eval_acc, 4), cur_epoch)

                    # save last checkpoint
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)

                    if args.data_num == -1 and args.save_last_checkpoints:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the last model into %s", output_model_file)

                    if eval_acc > best_acc:
                        not_acc_inc_cnt = 0
                        logger.info("  Best acc: %s", round(eval_acc, 4))
                        logger.info("  " + "*" * 20)
                        fa.write("[%d] Best acc changed into %.4f\n" % (cur_epoch, round(eval_acc, 4)))
                        best_acc = eval_acc
                        # Save best checkpoint for best ppl
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-acc')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        if args.data_num == -1 or True:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Save the best ppl model into %s", output_model_file)
                    else:
                        not_acc_inc_cnt += 1
                        logger.info("acc does not increase for %d epochs", not_acc_inc_cnt)
                        if not_acc_inc_cnt > args.patience:
                            logger.info("Early stop as acc do not increase for %d times", not_acc_inc_cnt)
                            fa.write("[%d] Early stop as not_acc_inc_cnt=%d\n" % (cur_epoch, not_acc_inc_cnt))
                            is_early_stop = True
                            break

                model.train()
            if is_early_stop:
                break

            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        for criteria in ['best-acc']:
            file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            logger.info("Reload model from {}".format(file))
            if hasattr(model, 'module'):
                model.module.load_state_dict(torch.load(file))
            else:
                model.load_state_dict(torch.load(file))

            # if args.n_gpu > 1:
            #     # multi-gpu training
            #     model = torch.nn.DataParallel(model)

            eval_examples, eval_data = load_and_cache_defect_data(args, args.test_filename, pool, tokenizer, 'test', False)

            result = evaluate(args, model, eval_examples, eval_data, write_to_pred=True)
            logger.info("  test_acc=%.4f", result['eval_acc'])
            logger.info("  " + "*" * 20)
            mv.add_metric('Test-Acc', result['eval_acc'])
            mv.add_metric('Test-Corrupt-Acc', result['eval_corrupt_acc'])
            mv.add_metric('Loss', result['eval_loss'])

            fa.write("[%s] test-acc: %.4f\n" % (criteria, result['eval_acc']))
            if args.res_file_name:
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)
                with open(os.path.join(args.output_dir, args.res_file_name), 'a+') as f:
                    f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
                    f.write("[%s] acc: %.4f\n\n" % (
                        criteria, result['eval_acc']))
    fa.close()


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


if __name__ == "__main__":
    # MV = MetricVisualizer('Origin')
    MV = MetricVisualizer('Our')
    for _ in range(5):
        main(MV)
    MV.summary()
    MV.traj_plot_by_trial()

    # MV = MetricVisualizer.load('Our-trial_id-0-metric_visualizer.dat')
    # MV = MetricVisualizer.load('Origin-trial_id-0-metric_visualizer.dat')
    # MV.summary()
import argparse
import json
import math
import os
import shutil
from typing import Type

import torch
import torch.distributed as dist
import transformers
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, BertModel, BertConfig
from transformers import BertTokenizer

from ner_task import sampling
from ner_task.entities import Dataset
from ner_task.evaluator import Evaluator
from ner_task.input_reader import JsonInputReader, BaseInputReader
from ner_task.loss import Loss, IdentifierLoss
from ner_task.trainer import BaseTrainer
from ner_task.util import to_device
from .model import MultiScaleSSN

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class PnRTrainer(BaseTrainer):
    """ Joint entity and relation extraction training and evaluation """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        # byte-pair encoding

        self._tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path,
                                                        local_files_only=True,
                                                        do_lower_case=args.lowercase,
                                                        cache_dir=args.cache_path)

        # path to export predictions to
        self._predictions_path = os.path.join(self._log_path, 'predictions_%s_epoch_%s.json')

        # path to export relation extraction examples to
        self._examples_path = os.path.join(self._log_path, 'examples_%s_%s_epoch_%s.html')

        self._logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

        self._logger.info("Save log at {}".format(self._log_path))
        shutil.copytree("pnr", os.path.join(
            self._log_path, "pnr")
        )  # backup code
        shutil.copytree("ner_task", os.path.join(
            self._log_path, "ner_task")
        )  # backup code
        log_dir, log_path_name = os.path.split(self._log_path)
        doc_dir = os.path.join(log_dir, "doc")
        if not os.path.exists(doc_dir):
            os.mkdir(doc_dir)
        with open(os.path.join(doc_dir, log_path_name + ".md"), "w") as f:
            # 创建文档文件
            pass

    def load_model(self, input_reader, is_eval=False):
        args = self.args
        embed = None
        if args.use_glove:
            embed = torch.from_numpy(input_reader.embedding_weight).float()

        config = BertConfig.from_pretrained(args.model_path, cache_dir=args.cache_path)
        if is_eval:
            # load model
            model = MultiScaleSSN.from_pretrained(
                args.model_path, cache_dir=args.cache_path,
                config=config,
                embed=embed,
                n_classes=input_reader.entity_type_count,
                n_queries=args.entity_queries_num,
                dropout=args.prop_drop,
                pool_type=args.pool_type,
                use_lstm=args.use_lstm,
                lstm_drop=args.lstm_drop,
                use_glove=args.use_glove,
                use_pos=args.use_pos,
                pos_size=args.pos_size,
                use_char_lstm=args.use_char_lstm,
                char_size=args.char_size,
                char_lstm_layers=args.char_lstm_layers,
                char_lstm_drop=args.char_lstm_drop,

                fpn_type=args.fpn_type,
                fpn_layer=args.fpn_layer,
                fpn_drop=args.prop_drop,
                use_topk_query=args.use_topk_query,
                use_msf=args.use_msf,
                dec_type=args.model_type,
                dec_layers=args.num_dec_layer,

                freeze_transformer=args.freeze_transformer,
                split_epoch=args.split_epoch,
                aux_loss=args.use_aux_loss,
            )
        else:
            # create model
            model = MultiScaleSSN(
                config=config,
                embed=embed,
                n_classes=input_reader.entity_type_count,
                n_queries=args.entity_queries_num,
                dropout=args.prop_drop,
                pool_type=args.pool_type,
                use_lstm=args.use_lstm,
                lstm_drop=args.lstm_drop,
                use_glove=args.use_glove,
                use_pos=args.use_pos,
                pos_size=args.pos_size,
                use_char_lstm=args.use_char_lstm,
                char_size=args.char_size,
                char_lstm_layers=args.char_lstm_layers,
                char_lstm_drop=args.char_lstm_drop,

                fpn_type=args.fpn_type,
                fpn_layer=args.fpn_layer,
                fpn_drop=args.prop_drop,
                use_topk_query=args.use_topk_query,
                use_msf=args.use_msf,
                dec_type=args.model_type,
                dec_layers=args.num_dec_layer,

                freeze_transformer=args.freeze_transformer,
                split_epoch=args.split_epoch,
                aux_loss=args.use_aux_loss,
            )

            if args.copy_weight and isinstance(model.encoder, BertModel):
                print("Use \"{}\" to initialize encoder of {}".format(
                    args.model_path, model.__class__.__name__
                ))
                model.encoder = BertModel.from_pretrained(args.model_path)
                model.freeze_bert_grad()

        # print(model)
        return model

    def train(self, train_path: str, valid_path: str, types_path: str, input_reader_cls: Type[BaseInputReader]):
        args = self.args
        train_label, valid_label = 'train', 'valid'

        if self.record:
            self._logger.info("Datasets: %s, %s" % (train_path, valid_path))
            self._logger.info("Model type: %s" % args.model_type)

            # create log csv files
            self._init_train_logging(train_label)
            self._init_eval_logging(valid_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer, self._logger, wordvec_filename=args.wordvec_path,
                                        random_mask_word=False, use_glove=args.use_glove,
                                        use_pos=args.use_pos, repeat_gt_entities=-1)
        input_reader.read({train_label: train_path, valid_label: valid_path})

        if self.local_rank < 1:
            self._log_datasets(input_reader)

        world_size = 1
        if args.local_rank != -1:
            world_size = dist.get_world_size()

        train_dataset = input_reader.get_dataset(train_label)
        train_sample_count = train_dataset.document_count
        updates_epoch = train_sample_count // (args.train_batch_size * world_size)
        updates_total = updates_epoch * args.split_epoch

        validation_dataset = input_reader.get_dataset(valid_label)

        if self.record:
            self._logger.info("Updates per epoch: %s" % updates_epoch)
            self._logger.info("Updates total: %s" % updates_total)

        model = self.load_model(input_reader, is_eval=False)

        model.to(self._device)
        if args.local_rank != -1:
            model = DDP(model, device_ids=[args.local_rank])

        # create optimizer
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay, correct_bias=False)
        # create scheduler
        # scheduler = transformers.get_constant_schedule_with_warmup(optimizer,
        #                                                     num_warmup_steps=args.lr_warmup * updates_total)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=args.lr_warmup * updates_total,
                                                                 num_training_steps=updates_total)
        # self.scheduler = scheduler
        # create loss function
        # entity_criterion = torch.nn.CrossEntropyLoss(reduction='none')

        compute_loss = IdentifierLoss(input_reader.entity_type_count, self._device, model, optimizer, scheduler,
                                      args.max_grad_norm, args.nil_weight, args.match_class_weight,
                                      args.match_boundary_weight, args.loss_class_weight, args.loss_boundary_weight,
                                      args.type_loss, solver=args.match_solver,
                                      match_warmup_epoch=args.match_warmup_epoch)

        # eval validation set
        if args.init_eval and self.record:
            self._eval(model, validation_dataset, input_reader, 0, updates_epoch)

        # train
        best_f1 = 0
        best_epoch = 0
        for epoch in range(args.epochs):
            if epoch == args.split_epoch:
                updates_total = updates_epoch * (args.epochs - args.split_epoch)
                # del compute_loss._optimizer
                # del compute_loss._scheduler
                # del optimizer
                # del scheduler
                # del compute_loss
                # torch.cuda.empty_cache()

                del optimizer
                # del compute_loss._optimizer
                del scheduler
                # del compute_loss._scheduler
                del compute_loss
                # torch.cuda.empty_cache()      # empty_cache会占用gpu 0
                optimizer = AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay, correct_bias=False)
                # del optimizer.state
                # optimizer.state = collections.defaultdict(dict)
                scheduler = transformers.get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps=args.lr_warmup * updates_total, num_training_steps=updates_total
                )
                # scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
                #     optimizer, num_warmup_steps=0.1 * updates_total, num_training_steps=updates_total, power=1.5
                # )
                compute_loss = IdentifierLoss(input_reader.entity_type_count, self._device, model, optimizer, scheduler,
                                              args.max_grad_norm, args.nil_weight, args.match_class_weight,
                                              args.match_boundary_weight, args.loss_class_weight,
                                              args.loss_boundary_weight, args.type_loss, solver=args.match_solver,
                                              match_warmup_epoch=args.match_warmup_epoch)

                # compute_loss._optimizer = optimizer
                # compute_loss._scheduler = self.scheduler

            # train epoch
            self._train_epoch(model, compute_loss, optimizer, train_dataset, updates_epoch, epoch)

            # eval validation sets
            if (not args.final_eval or (epoch == args.epochs - 1)) and self.record:
                f1 = self._eval(model, validation_dataset, input_reader, epoch + 1, updates_epoch)
                if best_f1 < f1[2]:
                    self._logger.info(f"Best F1 score update, from {best_f1} to {f1[2]}")
                    best_f1 = f1[2]
                    best_epoch = epoch + 1
                    extra = dict(epoch=epoch, updates_epoch=updates_epoch, epoch_iteration=0)
                    # if "pretrain" in args.label:
                    self._save_model(self._save_path, model, self._tokenizer, epoch * updates_epoch,
                                     optimizer=optimizer if args.save_optimizer else None, extra=extra,
                                     include_iteration=False, name='best_model')
            if self.record:
                self._logger.info(f"Best F1 score: {best_f1}, achieved at Epoch: {best_epoch}")

        # save final model
        extra = dict(epoch=args.epochs, updates_epoch=updates_epoch, epoch_iteration=0)
        global_iteration = args.epochs * updates_epoch
        if self.record:
            self._save_model(self._save_path, model, self._tokenizer, global_iteration,
                             optimizer=optimizer if args.save_optimizer else None, extra=extra,
                             include_iteration=False, name='final_model')
            self._logger.info("Logged in: %s" % self._log_path)
            self._logger.info("Saved in: %s" % self._save_path)
            self._close_summary_writer()

    def eval(self, dataset_path: str, types_path: str, input_reader_cls: Type[BaseInputReader]):
        args = self.args
        dataset_label = 'test'

        self._logger.info("Dataset: %s" % dataset_path)
        self._logger.info("Model: %s" % args.model_type)

        # create log csv files
        self._init_eval_logging(dataset_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer, self._logger, wordvec_filename=args.wordvec_path,
                                        random_mask_word=False, use_glove=args.use_glove,
                                        use_pos=args.use_pos, repeat_gt_entities=-1)
        input_reader.read({dataset_label: dataset_path})
        self._log_datasets(input_reader)

        model = self.load_model(input_reader, is_eval=True)

        model.to(self._device)
        # if args.local_rank != -1:
        #     model = DDP(model, device_ids=[args.local_rank])

        # evaluate
        self._eval(model, input_reader.get_dataset(dataset_label), input_reader)

        self._logger.info("Logged in: %s" % self._log_path)
        self._close_summary_writer()

    def _train_epoch(self, model: torch.nn.Module, compute_loss: Loss, optimizer: Optimizer, dataset,
                     updates_epoch: int, epoch: int):
        args = self.args
        self._logger.info("Train epoch: %s" % epoch)

        # create data loader
        dataset.switch_mode(Dataset.TRAIN_MODE)

        word_size = 1
        if args.local_rank != -1:
            word_size = dist.get_world_size()

        train_sampler = None
        shuffle = False
        if isinstance(dataset, Dataset):
            if len(dataset) < 100000:
                shuffle = True
            if args.local_rank != -1:
                train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=word_size,
                                                                                rank=args.local_rank, shuffle=shuffle)
                shuffle = False

        data_loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=shuffle, drop_last=True,
                                 num_workers=args.sampling_processes, collate_fn=sampling.collate_fn_padding,
                                 sampler=train_sampler)

        model.zero_grad()

        iteration = 0
        total = math.ceil((dataset.document_count // args.train_batch_size) / word_size)
        with tqdm(data_loader, total=total, ncols=0, desc='Train epoch %s' % epoch) as pbar:
            for batch in pbar:
                model.train()
                batch = to_device(batch, self._device)

                iteration += 1
                global_iteration = epoch * updates_epoch + iteration

                # forward step
                output = model.forward(encodings=batch['encodings'], context_masks=batch['context_masks'],
                                       seg_encoding=batch['seg_encoding'],
                                       context2token_masks=batch['context2token_masks'],
                                       token_masks=batch['token_masks'], epoch=epoch,
                                       pos_encoding=batch['pos_encoding'],
                                       wordvec_encoding=batch['wordvec_encoding'], char_encoding=batch['char_encoding'],
                                       token_masks_char=batch['token_masks_char'], char_count=batch['char_count'])
                # if model.two_stage and model.aux_loss:
                #    box_aux = output[0]
                #    output = output[1:]
                p_output = output[-1]
                p_entity, p_left, p_right = p_output["entity_logits"], p_output["p_left"], p_output["p_right"]

                # compute loss and optimize parameters
                batch_loss = compute_loss.compute(p_entity, p_left, p_right, output,
                                                  gt_types=batch['gt_types'], gt_spans=batch['gt_spans'],
                                                  entity_masks=batch['entity_masks'], epoch=epoch,
                                                  deeply_weight=args.deeply_weight,
                                                  gt_seq_labels=batch['gt_seq_labels'],
                                                  batch=batch)
                pbar.set_postfix(batch_loss="{:.3}".format(batch_loss / self.args.train_batch_size))
                # logging
                if global_iteration % args.train_log_iter == 0 and self.local_rank < 1:
                    self._log_train(optimizer, batch_loss, epoch, iteration, global_iteration, dataset.label)

        return iteration

    def _eval(self, model: torch.nn.Module, dataset, input_reader: JsonInputReader,
              epoch: int = 0, updates_epoch: int = 0, iteration: int = 0):
        args = self.args
        self._logger.info("Evaluate: %s" % dataset.label)

        # if isinstance(model, DataParallel):
        #     # currently no multi GPU support during evaluation
        #     model = model.module

        # create evaluator
        evaluator = Evaluator(dataset, input_reader, self._tokenizer, self._logger, args.no_overlapping,
                              args.no_partial_overlapping, args.no_duplicate, self._predictions_path,
                              self._examples_path, args.example_count, epoch, dataset.label,
                              cls_threshold=args.cls_threshold, boundary_threshold=args.boundary_threshold,
                              save_prediction=args.store_predictions)

        # create data loader
        dataset.switch_mode(Dataset.EVAL_MODE)

        word_size = 1
        eval_sampler = None

        if isinstance(dataset, Dataset):
            data_loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, drop_last=False,
                                     num_workers=args.sampling_processes, collate_fn=sampling.collate_fn_padding,
                                     sampler=eval_sampler)
        else:
            data_loader = DataLoader(dataset, batch_size=args.eval_batch_size, drop_last=False,
                                     collate_fn=sampling.collate_fn_padding, sampler=eval_sampler)

        dump_data = {
            "attn": [],
            "topk_indexes": [],
            "proposal_cls": [],
            "proposal_left": [],
            "proposal_right": [],
        }
        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(dataset.document_count / (args.eval_batch_size * word_size))
            for batch in tqdm(data_loader, total=total, ncols=0, desc='Evaluate epoch %s' % epoch):
                # move batch to selected device
                batch = to_device(batch, self._device)

                if self.args.label == "ms_attn_map":
                    # 获取attention map用于分析
                    piggyback = {
                        "attn": None,
                        "topk_indexes": None,
                        # (cls (bsz, n_q, n_ty), left (bsz, n_q, sent_len), right (bsz, n_q, sent_len))
                        "proposals": None,
                    }
                else:
                    piggyback = None
                # run model (forward pass)
                outputs = model.forward(
                    encodings=batch['encodings'], context_masks=batch['context_masks'],
                    seg_encoding=batch['seg_encoding'],
                    context2token_masks=batch['context2token_masks'],
                    token_masks=batch['token_masks'], pos_encoding=batch['pos_encoding'],
                    wordvec_encoding=batch['wordvec_encoding'],
                    char_encoding=batch['char_encoding'],
                    token_masks_char=batch['token_masks_char'], char_count=batch['char_count'],
                    evaluate=True, piggyback=piggyback
                )
                p_output = outputs[-1]
                p_entity, p_left, p_right = p_output["entity_logits"], p_output["p_left"], p_output["p_right"]

                # evaluate batch
                evaluator.eval_batch(p_entity, p_left, p_right, outputs, batch)

                if piggyback is not None:
                    for attn in piggyback["attn"]:
                        dump_data["attn"].append(attn.numpy())
                    for indexes in piggyback["topk_indexes"]:
                        dump_data["topk_indexes"].append(indexes.numpy())
                    proposals_cls, proposals_left, proposals_right = piggyback["proposals"]
                    for proposal_cls in proposals_cls:
                        dump_data["proposal_cls"].append(proposal_cls.numpy())
                    for proposal_left in proposals_left:
                        dump_data["proposal_left"].append(proposal_left.numpy())
                    for proposal_right in proposals_right:
                        dump_data["proposal_right"].append(proposal_right.numpy())

        global_iteration = epoch * updates_epoch + iteration
        ner_eval, ner_loc_eval, ner_cls_eval = evaluator.compute_scores()
        self._log_eval(*ner_eval, *ner_loc_eval, *ner_cls_eval, epoch, iteration, global_iteration, dataset.label)

        # self.scheduler.step(ner_eval[2])

        if args.store_predictions:
            evaluator.store_predictions()

        if args.store_examples:
            evaluator.store_examples()

        if self.args.label == "ms_attn_map":
            # dump data
            dump_path = os.path.join(self._log_path, "misc.pkl")
            torch.save(dump_data, dump_path)
            self._logger.info("Dump data at {}".format(dump_path))

        return ner_eval

    def _get_optimizer_params(self, model, bert_fine_tune_lr=2e-5):
        if self.args.copy_weight:
            # 考虑给fine-tuning的bert编码器的lr小一点
            bert_param_id_lst = list(map(id, model.encoder.parameters()))
            other_params = filter(lambda p: id(p[1]) not in bert_param_id_lst, model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_params = [
                {'params': model.encoder.parameters(), 'lr': bert_fine_tune_lr},
                {'params': [p for n, p in other_params if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.args.weight_decay},
                {'params': [p for n, p in other_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        else:
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_params = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.args.weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        return optimizer_params

    def _log_train(self, optimizer: Optimizer, loss: float, epoch: int,
                   iteration: int, global_iteration: int, label: str):
        # average loss
        avg_loss = loss / self.args.train_batch_size
        # get current learning rate
        lr = self._get_lr(optimizer)[-1]

        # log to tensorboard
        self._log_tensorboard(label, 'loss', loss, global_iteration)
        self._log_tensorboard(label, 'loss_avg', avg_loss, global_iteration)
        self._log_tensorboard(label, 'lr', lr, global_iteration)

        # log to csv
        self._log_csv(label, 'loss', loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'loss_avg', avg_loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'lr', lr, epoch, iteration, global_iteration)

    def _log_eval(self, ner_prec_micro: float, ner_rec_micro: float, ner_f1_micro: float,
                  ner_prec_macro: float, ner_rec_macro: float, ner_f1_macro: float,
                  loc_prec_micro: float, loc_rec_micro: float, loc_f1_micro: float,
                  loc_prec_macro: float, loc_rec_macro: float, loc_f1_macro: float,
                  cls_prec_micro: float, cls_rec_micro: float, cls_f1_micro: float,
                  cls_prec_macro: float, cls_rec_macro: float, cls_f1_macro: float,
                  epoch: int, iteration: int, global_iteration: int, label: str):

        # log to tensorboard
        self._log_tensorboard(label, 'eval/ner_prec_micro', ner_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_micro', ner_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_micro', ner_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_prec_macro', ner_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_macro', ner_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_macro', ner_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/loc_prec_micro', loc_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/loc_recall_micro', loc_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/loc_f1_micro', loc_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/loc_prec_macro', loc_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/loc_recall_macro', loc_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/loc_f1_macro', loc_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/cls_prec_micro', cls_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/cls_recall_micro', cls_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/cls_f1_micro', cls_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/cls_prec_macro', cls_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/cls_recall_macro', cls_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/cls_f1_macro', cls_f1_macro, global_iteration)

        # log to csv
        self._log_csv(label, 'eval', ner_prec_micro, ner_rec_micro, ner_f1_micro,
                      ner_prec_macro, ner_rec_macro, ner_f1_macro,
                      loc_prec_micro, loc_rec_micro, loc_f1_micro,
                      loc_prec_macro, loc_rec_macro, loc_f1_macro,
                      cls_prec_micro, cls_rec_micro, cls_f1_micro,
                      cls_prec_macro, cls_rec_macro, cls_f1_macro,
                      epoch, iteration, global_iteration)

    def _log_datasets(self, input_reader):
        # self._logger.info("Relation type count: %s" % input_reader.relation_type_count)
        self._logger.info("Entity type count: %s" % input_reader.entity_type_count)

        self._logger.info("Entities:")
        for e in input_reader.entity_types.values():
            self._logger.info(e.verbose_name + '=' + str(e.index))

        # self._logger.info("Relations:")
        # for r in input_reader.relation_types.values():
        #     self._logger.info(r.verbose_name + '=' + str(r.index))

        for k, d in input_reader.datasets.items():
            self._logger.info('Dataset: %s' % k)
            self._logger.info("Document count: %s" % d.document_count)
            # self._logger.info("Relation count: %s" % d.relation_count)
            self._logger.info("Entity count: %s" % d.entity_count)

        self._logger.info("Context size: %s" % input_reader.context_size)

    def _init_train_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'lr': ['lr', 'epoch', 'iteration', 'global_iteration'],
                                        'loss': ['loss', 'epoch', 'iteration', 'global_iteration'],
                                        'loss_avg': ['loss_avg', 'epoch', 'iteration', 'global_iteration']})

    def _init_eval_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'eval': ['ner_prec_micro', 'ner_rec_micro', 'ner_f1_micro',
                                                 'ner_prec_macro', 'ner_rec_macro', 'ner_f1_macro',
                                                 'loc_prec_micro', 'loc_rec_micro', 'loc_f1_micro',
                                                 'loc_prec_macro', 'loc_rec_macro', 'loc_f1_macro',
                                                 'cls_prec_micro', 'cls_rec_micro', 'cls_f1_micro',
                                                 'cls_prec_macro', 'cls_rec_macro', 'cls_f1_macro',
                                                 'epoch', 'iteration', 'global_iteration']})

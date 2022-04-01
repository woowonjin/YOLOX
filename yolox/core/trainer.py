#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import time
from loguru import logger
import copy

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import wandb
import sys
paths = os.getcwd().split("/")[0:-1]
base_path = "/".join(paths)
nets_path = os.path.join(base_path, "netspresso-compression-toolkit")
sys.path.append(nets_path)
from yolox.data import DataPrefetcher
from yolox.utils import (
    MeterBuffer,
    ModelEMA,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize
)
from .retrain_utils import RetrainUtils
from yolox.utils.raw_metrics import RawMetrics

class Trainer:
    def __init__(self, exp, args, mode="train"):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args
        if self.args.model:
            self.exp.model = torch.load(self.args.model)
        self.exp.basic_lr_per_img /= (10**self.args.lr_ratio)
        self.previous_lr = self.exp.basic_lr_per_img
        self.mode = mode
        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema # True

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0

        self.criteria = RetrainUtils(self.data_type)

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def finetune_lr(self):
        self.before_train()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        default_model = copy.deepcopy(self.model)
        default_model.eval()
        after_tune_acc = 0
        acceptable_deterioration = 0.01
        criterion = copy.deepcopy(self.criteria)
        optimizer = copy.deepcopy(self.optimizer)
        scheduler = copy.deepcopy(self.lr_scheduler)
        ap50_95, b4fine_tune_acc, summary = self.exp.eval(
            default_model, self.evaluator, self.is_distributed
        )
        b4fine_tune_acc = b4fine_tune_acc - acceptable_deterioration
        while (after_tune_acc < (b4fine_tune_acc)):
            self.model = copy.deepcopy(default_model).to(device)
            self.train_in_epoch()
            _, after_tune_acc, summary = self.exp.eval(
                self.model, self.evaluator, self.is_distributed
            )
            if (after_tune_acc < b4fine_tune_acc):
                self.exp.basic_lr_per_img = self.exp.basic_lr_per_img/10
                self.lr_scheduler = self.exp.get_lr_scheduler(
                    self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
                )
                self.criteria = criterion
                self.optimizer = optimizer
                scheduler = copy.deepcopy(self.lr_scheduler)
            else:
                self.criteria = criterion
                self.optimizer = optimizer
                self.lr_scheduler = scheduler
                self.model = copy.deepcopy(default_model)
                self.mode = "train"
                return ap50_95, b4fine_tune_acc, self.exp.basic_lr_per_img
        self.criteria = criterion
        self.optimizer = optimizer
        self.lr_scheduler = scheduler
        self.model = copy.deepcopy(default_model)
        self.mode = "train"
        return ap50_95, b4fine_tune_acc, self.exp.basic_lr_per_img

    def train(self):
        if self.args.use_wandb:
            if self.is_distributed:
                wandb.init(project="Nota-YOLOX", group=self.args.run_name, config=self.args)
            else:
                wandb.init(project="Nota-YOLOX", name=self.args.run_name, config=self.args)
        if self.mode == "optimize_lr":
            print("="*100)
            print("optimize_lr mode is True")
            print("="*100)
            ap50_95, ap50, finetuned_lr = self.finetune_lr()
            if self.args.use_wandb:
                wandb.log({"ap50_95": ap50_95, "ap50": ap50}, step=0)
            print("="*100)
            print("finetune_lr finished !!")
            print("="*100)
            if torch.cuda.is_available():
                print("="*100)
                print("cuda empty cache !!!")
                print("="*100)
                torch.cuda.empty_cache()
        else:
            print("="*100)
            print("optimize_lr mode is False")
            print("="*100)
            self.before_train()
            ap50_95, ap50, summary = self.exp.eval(
                self.model, self.evaluator, self.is_distributed
            )
            if self.args.use_wandb:
                wandb.log({"ap50_95": ap50_95, "ap50": ap50}, step=0)

        print("="*100)
        print(f"Learning Rate : {self.exp.basic_lr_per_img}")
        print("="*100)

        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.model.train()
            self.train_in_iter()
            if self.mode == "optimize_lr":
                break
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def train_one_iter(self):
        iter_start_time = time.time()
        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size) # scaling 작업
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            preds = self.model(inps)
            outputs = self.criteria(preds, targets, inps)
        loss = outputs["total_loss"]
        if self.mode == "train" and self.args.use_wandb:
            wandb.log({"loss": loss}, step=self.epoch+1)
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        self.previous_lr = lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        if self.args.model:
            model = self.exp.model
        else:
            model = self.exp.get_model()
        logger.info(
            "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        )
        model.to(self.device)
        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model
        self.model.train()

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )
        # Tensorboard logger
        if self.rank == 0:
            self.tblogger = SummaryWriter(self.file_name)

        logger.info("Training start...")
        logger.info("\n{}".format(model))

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100)
        )

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                self.criteria.use_l1 = False
                # self.model.module.head.use_l1 = True
            else:
                self.criteria.use_l1 = False
                # self.model.head.use_l1 = True
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )
            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )
            self.meter.clear_meters()

        # random resizing
        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            # resume the training states variables
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        ap50_95, ap50, summary = self.exp.eval(
            evalmodel, self.evaluator, self.is_distributed
        )
            
        raw_metrics = RawMetrics()
        raw_metrics_res = raw_metrics.get_raw_metrics(
            model=evalmodel,
            nms_thr=0.45, 
            score_thr=0.4, 
            iou_thr=0.5
        )# raw metrics 로깅을 위한 함수
        
        self.model.train()
        if self.rank == 0:
            self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
            self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            logger.info("\n" + summary)
        synchronize()
        if self.mode == "train" and self.args.use_wandb:
            wandb.log({"ap50_95": ap50_95, "ap50": ap50}, step=self.epoch+1)
            
            for metric in raw_metrics_res.keys():
                wandb.log({metric: raw_metrics_res[metric]}, step=self.epoch+1)
                print(f"{metric}: {raw_metrics_res[metric]}")

        self.save_ckpt("last_epoch", ap50_95 > self.best_ap)
        self.best_ap = max(self.best_ap, ap50_95)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank == 0:
            save_model = self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )

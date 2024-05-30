# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from dataclasses import asdict, dataclass, field
import os
import sys
import datetime
import time
import math
import json
import hydra.conf
import hydra.core
import numpy as np
import hydra.conf.hydra
import hydra.conf.hydra.env
import src
import src.models
from src.models.medsam import MedSAMIBot
import src.models.wrappers
import src.utils as utils
import src.models as models
from src.models.medsam import MedSAMIBot
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import hydra

from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
import wandb
from src.models.head import iBOTHead
from loader import ImageFolderMask
from evaluation.unsupervised.unsup_cls import eval_pred
from timm.models import resnet
import rich
from omegaconf import OmegaConf
import logging
from src.probing_nct import NCTProbing
import dotenv
from src.transform import DataAugmentation
import hydra
from src.ssl_evaluation import build_linear_probe_for_nct_patches


dotenv.load_dotenv()


def train_ibot(conf):
    # OmegaConf.save(conf, os.path.join(conf.output_dir, "config.yaml"), resolve=True)

    utils.init_distributed_mode(conf)
    utils.fix_random_seeds(conf.seed)

    if utils.is_main_process():
        logging.basicConfig(
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(os.path.join(conf.output_dir, "experiment.log")),
            ],
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        logging.getLogger("root").setLevel(logging.ERROR)

    # torch.backends.cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentation(
        **conf.transform,
    )

    pred_size = conf.patch_size * 8 if "swin" in conf.arch else conf.patch_size
    dataset = ImageFolderMask(
        conf.data_path,
        transform=transform,
        patch_size=pred_size,
        pred_ratio=conf.pred_ratio,
        pred_ratio_var=conf.pred_ratio_var,
        pred_aspect_ratio=(0.3, 1 / 0.3),
        pred_shape=conf.pred_shape,
        pred_start_epoch=conf.pred_start_epoch,
    )
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=conf.batch_size_per_gpu,
        num_workers=conf.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logging.info(f"Data loaded: there are {len(dataset)} images.")

    student, teacher = build_models(conf)

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = (
            nn.parallel.DistributedDataParallel(
                teacher, device_ids=[conf.gpu], broadcast_buffers=False
            )
            if "swin" in conf.arch
            else nn.parallel.DistributedDataParallel(teacher, device_ids=[conf.gpu])
        )
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = (
        nn.parallel.DistributedDataParallel(
            student, device_ids=[conf.gpu], broadcast_buffers=False
        )
        if "swin" in conf.arch
        else nn.parallel.DistributedDataParallel(student, device_ids=[conf.gpu])
    )
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    logging.info(f"Student and Teacher are built: they are both {conf.arch} network.")

    # ============ preparing loss ... ============
    same_dim = conf.shared_head or conf.shared_head_teacher

    ibot_loss = iBOTLoss(
        conf.out_dim,
        conf.out_dim if same_dim else conf.patch_out_dim,
        conf.transform.global_crops_number,
        conf.transform.local_crops_number,
        conf.warmup_teacher_temp,
        conf.teacher_temp,
        conf.warmup_teacher_patch_temp,
        conf.teacher_patch_temp,
        conf.warmup_teacher_temp_epochs,
        conf.epochs,
        lambda1=conf.lambda1,
        lambda2=conf.lambda2,
        mim_start_epoch=conf.pred_start_epoch,
    ).cuda()

    if utils.is_main_process():  # wandb
        wandb.init(
            project="iBOT",
            config=OmegaConf.to_container(conf, resolve=True),
            resume="allow",
            **conf.wandb
        )

    optimizer, lr_schedulers, wd_schedulers, momentum_schedule, fp16_scaler = (
        setup_optimization(student, len(data_loader), conf)
    )

    logging.info(f"Loss, optimizer and schedulers ready.")

    # ============ setup additional evaluation ===============
    ssl_evaluator = SSLEvaluator(conf)

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    restore_variables = {}
    restore_variables["student"] = student
    restore_variables["teacher"] = teacher

    # prefer to resume from latest checkpoint in output directory
    if os.path.exists(p := os.path.join(conf.output_dir, "checkpoint.pth")):
        logging.info(f"Loading state from {p}")
        utils.restart_from_checkpoint(
            p,
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            ibot_loss=ibot_loss,
        )
    elif (load_path := conf.load_from) is not None:
        # otherwise use 'load_from'
        assert os.path.exists(load_path), f"Load path {load_path} does not exist."
        logging.info(f"Loading state from {load_path}")
        utils.restart_from_checkpoint(
            load_path,
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            ibot_loss=ibot_loss,
        )
    elif (p := conf.load_model_from) is not None:
        assert os.path.exists(p), f"Load path {p} does not exist."
        logging.info(f"Loading models from {p}")
        utils.restart_from_checkpoint(
            p, run_variables=None, student=student, teacher=teacher, ibot_loss=ibot_loss
        )

    start_epoch = to_restore["epoch"]

    start_time = time.time()
    logging.info("Starting iBOT training!")
    for epoch in range(start_epoch, conf.epochs):
        data_loader.sampler.set_epoch(epoch)
        data_loader.dataset.set_epoch(epoch)

        # ============ training one epoch of iBOT ... ============
        logging.info(f"EPOCH {epoch}")
        train_stats = train_one_epoch(
            student,
            teacher,
            teacher_without_ddp,
            ibot_loss,
            data_loader,
            optimizer,
            lr_schedulers,
            wd_schedulers,
            momentum_schedule,
            epoch,
            fp16_scaler,
            conf,
        )

        # ============ writing logs ... ============
        save_dict = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": conf,
            "ibot_loss": ibot_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict["fp16_scaler"] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(conf.output_dir, "checkpoint.pth"))
        if conf.saveckp_freq and (epoch % conf.saveckp_freq == 0) and epoch:
            utils.save_on_master(
                save_dict, os.path.join(conf.output_dir, f"checkpoint{epoch:04}.pth")
            )

        # ======== run and log probing results ========
        if (epoch + 1) % conf.probing_freq == 0:
            probing_results = ssl_evaluator(
                teacher_without_ddp,
                epoch,
                "cuda",
                is_main_process=utils.is_main_process(),
            )
        else:
            probing_results = None
        if probing_results is not None:
            logging.info(f"Probing results: {probing_results}")
            wandb.log({"epoch": epoch, **probing_results})

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Training time {}".format(total_time_str))


def train_one_epoch(
    student,
    teacher,
    teacher_without_ddp: nn.Module,
    ibot_loss,
    data_loader,
    optimizer,
    lr_schedulers: list[list[float]],
    wd_schedulers: list[list[float]],
    momentum_schedule,
    epoch,
    fp16_scaler,
    conf,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, conf.epochs)

    # common params
    names_q, params_q, names_k, params_k = [], [], [], []
    for name_q, param_q in student.module.named_parameters():
        names_q.append(name_q)
        params_q.append(param_q)
    for name_k, param_k in teacher_without_ddp.named_parameters():
        names_k.append(name_k)
        params_k.append(param_k)
    names_common = list(set(names_q) & set(names_k))
    params_q = [
        param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common
    ]
    params_k = [
        param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common
    ]

    pred_labels, real_labels = [], []
    for it, (images, labels, masks) in enumerate(
        metric_logger.log_every(data_loader, 10, header)
    ):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        lrs_and_wds_for_logging = {}
        for i, (param_group, lr_schedule, wd_schedule) in enumerate(
            zip(optimizer.param_groups, lr_schedulers, wd_schedulers)
        ):
            param_group["lr"] = lr_schedule[it]
            param_group["weight_decay"] = wd_schedule[it]
            lrs_and_wds_for_logging.update(
                {
                    f"lr_{i}": lr_schedule[it],
                    f"wd_{i}": wd_schedule[it],
                }
            )

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        masks = [msk.cuda(non_blocking=True) for msk in masks]

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # get global views
            teacher_output = teacher(images[: conf.transform.global_crops_number])
            student_output = student(
                images[: conf.transform.global_crops_number],
                mask=masks[: conf.transform.global_crops_number],
            )

            # get local views
            student.module.backbone.masked_im_modeling = False
            student_local_cls = (
                student(images[conf.transform.global_crops_number :])[0]
                if len(images) > conf.transform.global_crops_number
                else None
            )
            student.module.backbone.masked_im_modeling = conf.use_masked_im_modeling

            all_loss = ibot_loss(
                student_output, teacher_output, student_local_cls, masks, epoch
            )
            loss = all_loss.pop("loss")

            if wandb.run is not None:
                wandb.log({"loss": loss.item(), **lrs_and_wds_for_logging})

        losses_across_batch = utils.concat_all_gather(loss.view(1))
        if any([not math.isfinite(loss_.item()) for loss_ in losses_across_batch]):
            logging.info("Warning: NaN value encountered in loss")
            continue

        # log statistics
        probs1 = teacher_output[0].chunk(conf.transform.global_crops_number)
        probs2 = student_output[0].chunk(conf.transform.global_crops_number)
        pred1 = utils.concat_all_gather(probs1[0].max(dim=1)[1])
        pred2 = utils.concat_all_gather(probs2[1].max(dim=1)[1])
        acc = (pred1 == pred2).sum() / pred1.size(0)
        pred_labels.append(pred1)
        real_labels.append(utils.concat_all_gather(labels.to(pred1.device)))

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if conf.clip_grad:
                param_norms = utils.clip_gradients(student, conf.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, conf.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if conf.clip_grad:
                fp16_scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, conf.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, conf.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(params_q, params_k):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        for key, value in all_loss.items():
            metric_logger.update(**{key: value.item()})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(acc=acc)

    pred_labels = torch.cat(pred_labels).cpu().detach().numpy()
    real_labels = torch.cat(real_labels).cpu().detach().numpy()

    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if conf.do_unsupervised_eval:
        nmi, ari, fscore, adjacc = eval_pred(real_labels, pred_labels, calc_acc=False)
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info(
            "NMI: {}, ARI: {}, F: {}, ACC: {}".format(nmi, ari, fscore, adjacc)
        )
        logging.info("Averaged stats:", metric_logger)
        return_dict.update({"nmi": nmi, "ari": ari, "fscore": fscore, "adjacc": adjacc})

    return return_dict


def build_models(conf):
    # ============ building student and teacher networks ... ============
    kw = dict(img_size=[conf.transform.global_crops_size])
    # we changed the name DeiT-S for ViT-S to avoid confusions
    conf.arch = conf.arch.replace("deit", "vit")
    # if the network is of hierechical features (i.e. swin_tiny, swin_small, swin_base)
    if conf.arch == "medsam":
        student = MedSAMIBot()
        teacher = MedSAMIBot()
        embed_dim = 768
    elif conf.arch in models.__dict__.keys() and "swin" in conf.arch:
        student = models.__dict__[conf.arch](
            window_size=conf.window_size,
            return_all_tokens=True,
            masked_im_modeling=conf.use_masked_im_modeling,
            **kw,
        )
        teacher = models.__dict__[conf.arch](
            window_size=conf.window_size,
            drop_path_rate=0.0,
            return_all_tokens=True,
            **kw,
        )
        embed_dim = student.num_features
    # if the network is a vision transformer (i.e. vit_tiny, vit_small, vit_base, vit_large)
    elif conf.arch in models.__dict__.keys():
        student = models.__dict__[conf.arch](
            patch_size=conf.patch_size,
            drop_path_rate=conf.drop_path,
            return_all_tokens=True,
            masked_im_modeling=conf.use_masked_im_modeling,
            **kw,
        )
        teacher = models.__dict__[conf.arch](
            patch_size=conf.patch_size, return_all_tokens=True, **kw
        )
        embed_dim = student.embed_dim
    elif "resnet" in conf.arch:
        student: resnet.ResNet = resnet.__dict__[conf.arch]()
        teacher = resnet.__dict__[conf.arch]()
        embed_dim = student.fc.weight.shape[1]
        student.fc = None
        teacher.fc = None
        student = src.models.wrappers.ResnetWrapper(student)
        teacher = src.models.wrappers.ResnetWrapper(teacher)
    else:
        logging.info(f"Unknow architecture: {conf.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(
        student,
        iBOTHead(
            embed_dim,
            conf.out_dim,
            patch_out_dim=conf.patch_out_dim,
            norm=conf.norm_in_head,
            act=conf.act_in_head,
            norm_last_layer=conf.norm_last_layer,
            shared_head=conf.shared_head,
        ),
    )
    teacher = utils.MultiCropWrapper(
        teacher,
        iBOTHead(
            embed_dim,
            conf.out_dim,
            patch_out_dim=conf.patch_out_dim,
            norm=conf.norm_in_head,
            act=conf.act_in_head,
            shared_head=conf.shared_head_teacher,
        ),
    )
    return student, teacher


def setup_optimization(student, num_iters_per_epoch, conf):
    batch_size_per_gpu = conf.batch_size_per_gpu
    epochs = conf.epochs

    def compute_true_lr_from_base_lr(base_lr):
        return base_lr * batch_size_per_gpu * utils.get_world_size() / 256.0

    params_groups = []
    lr_schedulers = []
    wd_schedulers = []

    for group_name, named_parameters in get_named_parameter_groups(student).items():

        opt_conf_for_group = conf.optimizer.get(group_name)
        if opt_conf_for_group is None:
            raise ValueError(
                f"Trying to configure optimizer for group {group_name}, but did not find it in config (keys {list(conf.optimizer.keys())})"
            )

        # get regularized vs. non-regularized parameter groups
        regularized = []
        not_regularized = []
        for name, param in named_parameters:
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)

        regularized = {"params": regularized}
        params_groups.append(regularized)
        not_regularized = {"params": not_regularized, "weight_decay": 0.0}
        params_groups.append(not_regularized)
        for _ in range(
            2
        ):  # lr has the same schedule for regularized vs. non_regularized
            lr_schedulers.append(
                utils.cosine_scheduler(
                    compute_true_lr_from_base_lr(
                        opt_conf_for_group.base_lr
                    ),  # linear scaling rule
                    opt_conf_for_group.min_lr,
                    epochs,
                    num_iters_per_epoch,
                    warmup_epochs=opt_conf_for_group.warmup_epochs,
                    frozen_epochs=opt_conf_for_group.frozen_epochs,
                    num_annealing_phases=opt_conf_for_group.get(
                        "num_annealing_phases", 1
                    ),
                )
            )
        wd_schedulers.append(
            utils.cosine_scheduler(
                opt_conf_for_group.weight_decay,
                opt_conf_for_group.weight_decay_end,
                epochs,
                num_iters_per_epoch,
            )
        )
        # second group receives 0 wd
        wd_schedulers.append(utils.cosine_scheduler(0, 0, epochs, num_iters_per_epoch))

    # ============ preparing optimizer ... ============
    if conf.optimizer.type == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif conf.optimizer.type == "sgd":
        optimizer = torch.optim.SGD(
            params_groups, lr=0, momentum=0.9
        )  # lr is set by scheduler
    elif conf.optimizer.type == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches

    # for mixed precision training
    fp16_scaler = None
    if conf.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(
        conf.momentum_teacher, 1, conf.epochs, num_iters_per_epoch
    )

    return optimizer, lr_schedulers, wd_schedulers, momentum_schedule, fp16_scaler


def get_named_parameter_groups(model: nn.Module):
    if isinstance(model, MedSAMIBot):
        raise NotImplementedError()
    else:
        return {"main": model.named_parameters()}


class iBOTLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        patch_out_dim,
        ngcrops,
        nlcrops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp2,
        teacher_temp2,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
        center_momentum2=0.9,
        lambda1=1.0,
        lambda2=1.0,
        mim_start_epoch=0,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.ngcrops = ngcrops
        self.nlcrops = nlcrops
        self.ncrops = ngcrops + nlcrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(
                    warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
                ),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )
        self.teacher_temp2_schedule = (
            np.concatenate(
                (
                    np.linspace(
                        warmup_teacher_temp2, teacher_temp2, warmup_teacher_temp_epochs
                    ),
                    np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp2,
                )
            )
            if mim_start_epoch == 0
            else np.concatenate(
                (
                    np.ones(mim_start_epoch) * warmup_teacher_temp2,
                    np.linspace(
                        warmup_teacher_temp2, teacher_temp2, warmup_teacher_temp_epochs
                    ),
                    np.ones(nepochs - warmup_teacher_temp_epochs - mim_start_epoch)
                    * teacher_temp2,
                )
            )
        )

    def forward(
        self, student_output, teacher_output, student_local_cls, student_mask, epoch
    ):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_cls, student_patch = student_output
        teacher_cls, teacher_patch = teacher_output

        if student_local_cls is not None:
            student_cls = torch.cat([student_cls, student_local_cls])

        # [CLS] and patch for global patches
        student_cls = student_cls / self.student_temp
        student_cls_c = student_cls.chunk(self.ncrops)
        student_patch = student_patch / self.student_temp
        student_patch_c = student_patch.chunk(self.ngcrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]
        teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)
        teacher_cls_c = teacher_cls_c.detach().chunk(self.ngcrops)
        teacher_patch_c = F.softmax((teacher_patch - self.center2) / temp2, dim=-1)
        teacher_patch_c = teacher_patch_c.detach().chunk(self.ngcrops)

        total_loss1, n_loss_terms1 = 0, 0
        total_loss2, n_loss_terms2 = 0, 0
        for q in range(len(teacher_cls_c)):
            for v in range(len(student_cls_c)):
                if v == q:
                    loss2 = torch.sum(
                        -teacher_patch_c[q] * F.log_softmax(student_patch_c[v], dim=-1),
                        dim=-1,
                    )
                    mask = student_mask[v].flatten(-2, -1)
                    loss2 = torch.sum(loss2 * mask.float(), dim=-1) / mask.sum(
                        dim=-1
                    ).clamp(min=1.0)
                    total_loss2 += loss2.mean()
                    n_loss_terms2 += 1
                else:
                    loss1 = torch.sum(
                        -teacher_cls_c[q] * F.log_softmax(student_cls_c[v], dim=-1),
                        dim=-1,
                    )
                    total_loss1 += loss1.mean()
                    n_loss_terms1 += 1

        total_loss1 = total_loss1 / n_loss_terms1 * self.lambda1
        total_loss2 = total_loss2 / n_loss_terms2 * self.lambda2
        total_loss = dict(
            cls=total_loss1, patch=total_loss2, loss=total_loss1 + total_loss2
        )
        self.update_center(teacher_cls, teacher_patch)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_patch):
        """
        Update center used for teacher output.
        """
        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        dist.all_reduce(cls_center)
        cls_center = cls_center / (len(teacher_cls) * dist.get_world_size())
        self.center = self.center * self.center_momentum + cls_center * (
            1 - self.center_momentum
        )

        patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
        dist.all_reduce(patch_center)
        patch_center = patch_center / (len(teacher_patch) * dist.get_world_size())
        self.center2 = self.center2 * self.center_momentum2 + patch_center * (
            1 - self.center_momentum2
        )


class SSLEvaluator:
    def __init__(self, conf):
        if conf.do_nct_probing:
            logging.info(f"Setting up NCT probing...")
            self.nct_probe = NCTProbing(
                conf.batch_size_per_gpu, size=conf.transform.global_crops_size
            )
        else:
            self.nct_probe = None

        if conf.get('do_clstoken_nct_probing', False):
            logging.info(f"Setting up patch NCT probing...")
            self.nct_clstoken_probe = build_linear_probe_for_nct_patches(
                input_size=conf.transform.global_crops_size,
                batch_size=conf.batch_size_per_gpu,
                device="cuda",
            )

    def __call__(self, model: utils.MultiCropWrapper, epoch, device, is_main_process):
        logging.info(f"Running NCT Probing")
        probing_results = {}

        if self.nct_probe is not None:
            train_metrics, val_metrics = self.nct_probe.run_probing(
                model,
                epoch,
                "cuda",
                is_main_process=utils.is_main_process(),
            )
            self._add_metrics_to_dict(
                probing_results, train_metrics, "train", "probing"
            )
            self._add_metrics_to_dict(probing_results, val_metrics, "val", "probing")

        if self.nct_clstoken_probe is not None:
            train_metrics, val_metrics = self.nct_clstoken_probe.run_probing(
                model
            )
            self._add_metrics_to_dict(
                probing_results, train_metrics, "train", "clstoken_probing"
            )
            self._add_metrics_to_dict(
                probing_results, val_metrics, "val", "clstoken_probing"
            )

        return probing_results

    def _add_metrics_to_dict(self, d, metrics, split, name):
        d.update({f"{split}_{k}_{name}": v for k, v in metrics.items()})


@hydra.main(config_path="conf", config_name="medsam_ibot.yaml", version_base=None)
def main(conf):
    if conf.get("wandb_path", None) is not None:
        api = wandb.Api()
        run = api.run(conf.wandb_path)
        conf = OmegaConf.create(run.config)
        logging.info(f"{run.id} loaded from wandb.")

    rich.print(OmegaConf.to_yaml(conf))
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    checkpoint_dir_symlink_target = os.path.join(output_dir, "checkpoint")
    checkpoint_dir = conf.output_dir
    if checkpoint_dir is None: 
        checkpoint_dir = output_dir 
        conf.output_dir = output_dir
    else: 
        checkpoint_dir = os.path.abspath(conf.output_dir)
        print(checkpoint_dir)
        if checkpoint_dir != checkpoint_dir_symlink_target:
            # sym link the output dir to the checkpoint dir
            try:
                os.symlink(
                    checkpoint_dir, checkpoint_dir_symlink_target, target_is_directory=True
                )
            except:
                pass

    train_ibot(conf)


if __name__ == "__main__":
    main()

    # parser = argparse.ArgumentParser(add_help=False)
    # parser.add_argument(
    #     "--config", "-c",
    # )
    # parser.add_argument(
    #     "--wandb_path", help='Name of wandb path (path to run in the form `entity/project/run_id`) - if set, will use this id to load the configuration from a wandb run and resume it.'
    # )
    #
    # parser.add_argument("--help", "-h", action="store_true")
    # args, extras = parser.parse_known_args()
#
# if args.wandb_path is not None:
#     api = wandb.Api()
#     run = api.run(args.wandb_path)
#     conf = OmegaConf.create(run.config)
#     conf.wandb_run_id = run.id
#
# elif args.config is not None:
#     conf = OmegaConf.load(args.config)
#     conf = OmegaConf.merge(conf, OmegaConf.from_cli(extras))
# else:
#     parser.print_help()
#     sys.exit(1)
#
# if args.help:
#     print("========")
#     parser.print_help()
#     print("========")
#     rich.print(OmegaConf.to_yaml(conf))
#     print("========")
#     sys.exit(0)
# #
# train_ibot(conf)
#

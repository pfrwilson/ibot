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
import numpy as np
import utils
import models
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
import wandb
from models.head import iBOTHead
from loader import ImageFolderMask
from evaluation.unsupervised.unsup_cls import eval_pred
from timm.models import resnet
import rich_argparse

import dotenv

dotenv.load_dotenv()


@dataclass
class AugmentationArgs: 
    """Args for DataAugmentationIBot"""
    global_crops_scale: tuple[float, float] = field(default_factory=lambda:(0.14, 1))
    local_crops_scale: tuple[float, float] = field(default_factory=lambda:(0.05, 0.4))
    global_crops_number: int = 2
    local_crops_number: int = 0
    global_crops_size: int = 224
    local_crops_size: int = 96
    jitter_prob: float = 0.8
    mean: list[float] = field(default_factory=lambda:[0.485, 0.456, 0.406])
    std: list[float] = field(default_factory=lambda:[0.229, 0.224, 0.225])
    blur_prob_1: float = 1.0
    blur_prob_2: float = 0.1
    solarization_prob: float = 0.2
    initial_crop_size: int = None
    initial_resize_size: int = None


@dataclass
class Args:
    """
    Args for main training.

    Args:
        arch (str): Name of architecture to train. For quick experiments with ViTs,
            we recommend using vit_tiny or vit_small.
        patch_size (int): Size in pixels of input square patches - default 16 (for 16x16 patches). 
            Using smaller values leads to better performance but requires more memory. 
            Applies only for ViTs (vit_tiny, vit_small and vit_base). 
            If <16, we recommend disabling mixed precision training (--use_fp16 false) to avoid unstabilities.
        window_size (int): Size of window - default 7. This config is only valid for Swin Transofmer 
            and is ignoired for vanilla ViT architectures.
        out_dim (int): Dimensionality of output for [CLS] token.
        patch_out_dim (int): Dimensionality of output for patch tokens.
        shared_head (bool): Whether to share the same head for [CLS] token output and patch tokens output. 
            When set to false, patch_out_dim is ignored and enforced to be same with out_dim. (Default: False)
        shared_head_teacher (bool): See above. Only works for teacher model. (Defeault: True)
        norm_last_layer (bool): Whether or not to weight normalize the last layer of the head.
            Not normalizing leads to better performance but can make the training unstable.
            In our experiments, we typically set this paramater to False with vit_small and True with vit_base.
        momentum_teacher (float): Base EMA parameter for teacher update. The value is increased to 1 during 
            training with cosine schedule. We recommend setting a higher value with small batches: 
            for example use 0.9995 with batch size of 256.
        norm_in_head (str): Whether to use batch normalizations in projection head (Default: None)
        act_in_head (str): Whether to use batch normalizations in projection head (Default: gelu)
        use_masked_im_modeling (bool): Whether to use masked image modeling (mim) in backbone (Default: True)
        pred_ratio (float): Ratio of partial prediction. If a list of ratio is specified, 
            one of them will be randomly choosed for each patch.
        pred_ratio_var (float): Variance of partial prediction ratio. Length should be indentical 
            to the length of pred_ratio. 0 for disabling.
        pred_shape (str): Shape of partial prediction.
        pred_start_epoch (int): Start epoch to perform masked image prediction. 
            We typically set this to 50 for swin transformer. (Default: 0)
        lambda1 (float): loss weight for dino loss over [CLS] tokens (Default: 1.0)
        lambda2 (float): loss weight for beit loss over masked patch tokens (Default: 1.0)
        warmup_teacher_temp (float): Initial value for the teacher temperature: 
            0.04 works well in most cases. Try decreasing it if the training loss does not decrease.
        teacher_temp (float): Final value (after linear warmup) of the teacher temperature. 
            For most experiments, anything above 0.07 is unstable. We recommend starting with 
            the default value of 0.04 and increase this slightly if needed.
        warmup_teacher_patch_temp (float): See `--warmup_teacher_temp`
        teacher_patch_temp (float): See `--teacher_temp`
        warmup_teacher_temp_epochs (int): Number of warmup epochs for the teacher temperature (Default: 30).
        use_fp16 (bool): Whether or not to use half precision for training. Improves training time and 
            memory requirements, but can provoke instability and slight decay of performance. 
            We recommend disabling mixed precision if the loss is unstable, 
            if reducing the patch size or if training with bigger ViTs.
        weight_decay (float): Initial value of the weight decay. With ViT, a smaller value at the 
            beginning of training works well.
        weight_decay_end (float): Final value of the weight decay. We use a cosine schedule for WD and 
            using a larger decay by the end of training improves performance for ViTs.
        clip_grad (float): Maximal parameter gradient norm if using gradient clipping. 
            Clipping with norm .3 ~ 1.0 can help optimization for larger ViT architectures. 0 for disabling.
        batch_size_per_gpu (int): Per-GPU batch-size : number of distinct images loaded on one GPU.
        epochs (int): Number of epochs of training.
        freeze_last_layer (int): Number of epochs during which we keep the output layer fixed. 
            Typically doing so during the first epoch helps training. Try increasing this value 
            if the loss does not decrease.
        lr (float): Learning rate at the end of linear warmup (highest LR used during training). 
            The learning rate is linearly scaled with the batch size, and specified here for a 
            reference batch size of 256.
        warmup_epochs (int): Number of epochs for the linear learning-rate warm up.
        min_lr (float): Target LR at the end of optimization. We use a cosine LR schedule with linear warmup.
        optimizer (str): Type of optimizer. We recommend using adamw with ViTs.
        load_from (str): Path to load checkpoints to resume training.
        load_model_from (str): Path to checkpoint to load *models* from. Unlike CKPT_PATH, only models 
            would be loaded from here, not other run state.
        drop_path (float): Drop path rate for student network.
        data_path (str): Please specify path to the ImageNet training data.
        output_dir (str): Path to save logs and checkpoints.
        saveckp_freq (int): Save checkpoint every x epochs.
        seed (int): Random seed.
        num_workers (int): Number of data loading workers per GPU.
        dist_url (str): url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html
        local_rank (int): Please ignore and do not set this argument.
        do_unsupervised_eval (bool): Whether to perform unsupervised evaluation, 
            (clustering cls token -> image folder labels)
        do_nct_probing (bool): Whether to perform probing on the NCT micro-ultrasound dataset
        probing_freq (int): Probing frequency (do probing every N epochs)
        wandb_run_id (str): WandB run ID
    """
    arch: str = 'vit_small'
    patch_size: int = 16
    window_size: int = 7
    out_dim: int = 8192
    patch_out_dim: int = 8192
    shared_head: bool = False
    shared_head_teacher: bool = True
    norm_last_layer: bool = True
    momentum_teacher: float = 0.996
    norm_in_head: str = None
    act_in_head: str = 'gelu'
    use_masked_im_modeling: bool = True
    pred_ratio: float = 0.3
    pred_ratio_var: float = 0
    pred_shape: str = 'block'
    pred_start_epoch: int = 0
    lambda1: float = 1.0
    lambda2: float = 1.0
    warmup_teacher_temp: float = 0.04
    teacher_temp: float = 0.04
    warmup_teacher_patch_temp: float = 0.04
    teacher_patch_temp: float = 0.07
    warmup_teacher_temp_epochs: int = 30
    use_fp16: bool = True
    weight_decay: float = 0.04
    weight_decay_end: float = 0.4
    clip_grad: float = 3.0
    batch_size_per_gpu: int = 128
    epochs: int = 100
    freeze_last_layer: int = 1
    lr: float = 0.0005
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    optimizer: str = 'adamw'
    load_from: str = None
    load_model_from: str = None
    drop_path: float = 0.1
    data_path: str = os.environ['DATA_PATH']
    output_dir: str = "."
    saveckp_freq: int = 40
    seed: int = 0
    num_workers: int = 10
    dist_url: str = "env://"
    local_rank: int = 0
    do_unsupervised_eval: bool = False
    do_nct_probing: bool = False
    probing_freq: int = 1
    wandb_run_id: str = None

    data_augmentation: AugmentationArgs = field(default_factory=AugmentationArgs)


def train_ibot(args: Args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)

    print("git:\n  {}\n".format(utils.get_sha()))
    print(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationiBOT(
        **asdict(args.data_augmentation),
    )

    pred_size = args.patch_size * 8 if "swin" in args.arch else args.patch_size
    dataset = ImageFolderMask(
        args.data_path,
        transform=transform,
        patch_size=pred_size,
        pred_ratio=args.pred_ratio,
        pred_ratio_var=args.pred_ratio_var,
        pred_aspect_ratio=(0.3, 1 / 0.3),
        pred_shape=args.pred_shape,
        pred_start_epoch=args.pred_start_epoch,
    )
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    kw = dict(
        img_size=[args.data_augmentation.global_crops_size]
    )
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is of hierechical features (i.e. swin_tiny, swin_small, swin_base)
    if args.arch in models.__dict__.keys() and "swin" in args.arch:
        student = models.__dict__[args.arch](
            window_size=args.window_size,
            return_all_tokens=True,
            masked_im_modeling=args.use_masked_im_modeling,
            **kw
        )
        teacher = models.__dict__[args.arch](
            window_size=args.window_size,
            drop_path_rate=0.0,
            return_all_tokens=True,
            **kw
        )
        embed_dim = student.num_features
    # if the network is a vision transformer (i.e. vit_tiny, vit_small, vit_base, vit_large)
    elif args.arch in models.__dict__.keys():
        student = models.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path,
            return_all_tokens=True,
            masked_im_modeling=args.use_masked_im_modeling,
            **kw
        )
        teacher = models.__dict__[args.arch](
            patch_size=args.patch_size,
            return_all_tokens=True,
            **kw
        )
        embed_dim = student.embed_dim
    elif "resnet" in args.arch:
        student: resnet.ResNet = resnet.__dict__[args.arch]()
        teacher = resnet.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
        student.fc = None
        teacher.fc = None
        student = utils.ResnetWrapper(student)
        teacher = utils.ResnetWrapper(teacher)
    elif args.arch == "medsam":
        from models.medsam import MedSAMIBot

        student = MedSAMIBot()
        teacher = MedSAMIBot()
        embed_dim = 768
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(
        student,
        iBOTHead(
            embed_dim,
            args.out_dim,
            patch_out_dim=args.patch_out_dim,
            norm=args.norm_in_head,
            act=args.act_in_head,
            norm_last_layer=args.norm_last_layer,
            shared_head=args.shared_head,
        ),
    )
    teacher = utils.MultiCropWrapper(
        teacher,
        iBOTHead(
            embed_dim,
            args.out_dim,
            patch_out_dim=args.patch_out_dim,
            norm=args.norm_in_head,
            act=args.act_in_head,
            shared_head=args.shared_head_teacher,
        ),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = (
            nn.parallel.DistributedDataParallel(
                teacher, device_ids=[args.gpu], broadcast_buffers=False
            )
            if "swin" in args.arch
            else nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        )
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = (
        nn.parallel.DistributedDataParallel(
            student, device_ids=[args.gpu], broadcast_buffers=False
        )
        if "swin" in args.arch
        else nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    )
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    same_dim = args.shared_head or args.shared_head_teacher

    ibot_loss = iBOTLoss(
        args.out_dim,
        args.out_dim if same_dim else args.patch_out_dim,
        args.data_augmentation.global_crops_number,
        args.data_augmentation.local_crops_number,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_patch_temp,
        args.teacher_patch_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        mim_start_epoch=args.pred_start_epoch,
    ).cuda()

    if utils.is_main_process():  # wandb
        run = wandb.init(
            project="iBOT", config=asdict(args), id=args.wandb_run_id, resume="allow"
        )

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params_groups, lr=0, momentum=0.9
        )  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr
        * (args.batch_size_per_gpu * utils.get_world_size())
        / 256.0,  # linear scaling rule
        args.min_lr,
        args.epochs,
        len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(
        args.momentum_teacher, 1, args.epochs, len(data_loader)
    )

    print(f"Loss, optimizer and schedulers ready.")

    # ============ setup additional evaluation ===============
    if args.do_nct_probing:
        print(f"Setting up NCT probing...")
        from probing_nct import NCTProbing

        nct_probe = NCTProbing(args.batch_size_per_gpu, size=args.data_augmentation.global_crops_size)

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    restore_variables = {}
    restore_variables["student"] = student
    restore_variables["teacher"] = teacher

    # prefer to resume from latest checkpoint in output directory
    if os.path.exists(p := os.path.join(args.output_dir, "checkpoint.pth")):
        print(f"Loading state from {p}")
        utils.restart_from_checkpoint(
            p,
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            ibot_loss=ibot_loss,
        )
    elif (load_path := args.load_from) is not None:
        # otherwise use 'load_from'
        assert os.path.exists(load_path), f"Load path {load_path} does not exist."
        print(f"Loading state from {load_path}")
        utils.restart_from_checkpoint(
            load_path,
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            ibot_loss=ibot_loss,
        )
    elif (p := args.load_model_from) is not None:
        assert os.path.exists(p), f"Load path {p} does not exist."
        print(f"Loading models from {p}")
        utils.restart_from_checkpoint(
            p, run_variables=None, student=student, teacher=teacher, ibot_loss=ibot_loss
        )

    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting iBOT training!")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        data_loader.dataset.set_epoch(epoch)

        # ============ training one epoch of iBOT ... ============
        train_stats = train_one_epoch(
            student,
            teacher,
            teacher_without_ddp,
            ibot_loss,
            data_loader,
            optimizer,
            lr_schedule,
            wd_schedule,
            momentum_schedule,
            epoch,
            fp16_scaler,
            args,
        )

        # ============ writing logs ... ============
        save_dict = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": args,
            "ibot_loss": ibot_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict["fp16_scaler"] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, "checkpoint.pth"))
        if args.saveckp_freq and (epoch % args.saveckp_freq == 0) and epoch:
            utils.save_on_master(
                save_dict, os.path.join(args.output_dir, f"checkpoint{epoch:04}.pth")
            )
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }

        # ======== run and log probing results ========
        if args.do_nct_probing and ((epoch + 1) % args.probing_freq == 0):
            print(f"Running NCT Probing")
            probing_results = nct_probe.run_probing(
                teacher_without_ddp,
                epoch,
                "cuda",
                is_main_process=utils.is_main_process(),
            )
            print(f"Probing results: {probing_results}")
        else:
            probing_results = None
        if probing_results is not None:
            log_stats.update(probing_results)

        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
                run.log(log_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def train_one_epoch(
    student,
    teacher,
    teacher_without_ddp,
    ibot_loss,
    data_loader,
    optimizer,
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    epoch,
    fp16_scaler,
    args: Args,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)

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
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        masks = [msk.cuda(non_blocking=True) for msk in masks]

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # get global views
            teacher_output = teacher(images[: args.data_augmentation.global_crops_number])
            student_output = student(
                images[: args.data_augmentation.global_crops_number],
                mask=masks[: args.data_augmentation.global_crops_number],
            )

            # get local views
            student.module.backbone.masked_im_modeling = False
            student_local_cls = (
                student(images[args.data_augmentation.global_crops_number :])[0]
                if len(images) > args.data_augmentation.global_crops_number
                else None
            )
            student.module.backbone.masked_im_modeling = args.use_masked_im_modeling

            all_loss = ibot_loss(
                student_output, teacher_output, student_local_cls, masks, epoch
            )
            loss = all_loss.pop("loss")

            if wandb.run is not None:
                wandb.log({"loss": loss.item()})

        losses_across_batch = utils.concat_all_gather(loss.view(1))
        if any([not math.isfinite(loss_.item()) for loss_ in losses_across_batch]):
            print("Warning: NaN value encountered in loss")
            continue

        # log statistics
        probs1 = teacher_output[0].chunk(args.data_augmentation.global_crops_number)
        probs2 = student_output[0].chunk(args.data_augmentation.global_crops_number)
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
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
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

    if args.do_unsupervised_eval:
        nmi, ari, fscore, adjacc = eval_pred(real_labels, pred_labels, calc_acc=False)
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("NMI: {}, ARI: {}, F: {}, ACC: {}".format(nmi, ari, fscore, adjacc))
        print("Averaged stats:", metric_logger)
        return_dict.update({"nmi": nmi, "ari": ari, "fscore": fscore, "adjacc": adjacc})

    return return_dict


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


class DataAugmentationiBOT(object):
    def __init__(
        self,
        global_crops_scale: list[float, float] = (0.14, 1),
        local_crops_scale: list[float, float] = (0.05, 0.4),
        global_crops_number: int = 2,
        local_crops_number: int = 0,
        global_crops_size: int = 224,
        local_crops_size: int = 96,
        jitter_prob: float = 0.8,
        mean: list[float] = [0.485, 0.456, 0.406],
        std: list[float] = [0.229, 0.224, 0.225],
        blur_prob_1: float = 1.0,
        blur_prob_2: float = 0.1,
        solarization_prob: float = 0.2,
        initial_crop_size: int = None,
        initial_resize_size: int = None,
    ):
        self.std = std
        self.mean = mean

        self.initial_crop = (
            transforms.RandomCrop(initial_crop_size)
            if initial_crop_size is not None
            else lambda x: x
        )
        self.initial_resize = (
            transforms.Resize(
                (initial_resize_size, initial_resize_size), antialias=True
            )
            if initial_resize_size is not None
            else lambda x: x
        )

        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=jitter_prob,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.global_crops_number = global_crops_number
        # transformation for the first global crop
        self.global_transfo1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size,
                    scale=global_crops_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_and_color_jitter,
                utils.GaussianBlur(p=blur_prob_1),
                normalize,
            ]
        )
        # transformation for the rest of global crops
        self.global_transfo2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size,
                    scale=global_crops_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_and_color_jitter,
                utils.GaussianBlur(p=blur_prob_1),
                utils.Solarization(solarization_prob),
                normalize,
            ]
        )
        # transformation for the local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_and_color_jitter,
                utils.GaussianBlur(p=blur_prob_2),
                normalize,
            ]
        )

    def __call__(self, image):
        image = self.initial_resize(image)
        image = self.initial_crop(image)
        crops = []
        crops.append(self.global_transfo1(image))
        for _ in range(self.global_crops_number - 1):
            crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

    def to_numpy(self, tensor):
        C, H, W = tensor.shape
        tensor *= torch.tensor(self.std)[..., None, None]
        tensor += torch.tensor(self.mean)[..., None, None]
        return tensor.permute(1, 2, 0).numpy()


if __name__ == "__main__":
    from simple_parsing import parse 
    args: Args = parse(Args)


    # parser = argparse.ArgumentParser(
    #     "iBOT",
    #     parents=[get_args_parser()],
    #     formatter_class=rich_argparse.ArgumentDefaultsRichHelpFormatter,
    # )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_ibot(args)

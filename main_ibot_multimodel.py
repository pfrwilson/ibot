import argparse
from itertools import chain
import os
import sys
import math
import src.utils as utils
from src.models.medsam import MedSAMIBot
import torch
import torch.nn as nn
import torch.distributed as dist
import wandb
from loader import ImageFolderMask
import rich
from omegaconf import DictConfig, OmegaConf
import logging
from src.probing_nct import NCTProbing
import dotenv
from src.transform import DataAugmentation
from src.ssl_evaluation import (
    build_linear_probe_for_nct_patches,
    build_kfold_linear_probe_for_nct_patches,
)
from timm.models.resnet import resnet18
from src.models.head import DINOHead
from src.loss import iBOTLoss, DINOLoss
from torch import distributed as dist
from src.models.base import BackboneNetwork
from src.argparse_utils import LoadOmegaConf, OverrideYaml
from rich_argparse import ArgumentDefaultsRichHelpFormatter


dotenv.load_dotenv()


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(
        self,
        backbone,
        head=None,
    ):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone

        if head is None:
            self.head = nn.Identity()
        else:
            self.head = head

    def forward(self, x, mask=None, return_backbone_feat=False, **kwargs):
        # convert to list
        if not isinstance(x, list):
            x = [x]
            mask = [mask] if mask is not None else None
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            0,
        )
        start_idx = 0
        for end_idx in idx_crops:
            inp_x = torch.cat(x[start_idx:end_idx])

            if mask is not None:
                inp_m = torch.cat(mask[start_idx:end_idx])
                kwargs.update(dict(mask=inp_m))

            _out = self.backbone(inp_x, **kwargs)
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        output_ = self.head(output)
        if return_backbone_feat:
            return output, output_
        return output_


def train_ibot(conf):

    # ========= setup =======================
    os.makedirs(conf.output_dir, exist_ok=True)
    OmegaConf.save(conf, os.path.join(conf.output_dir, "config.yaml"), resolve=True)

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())
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
        conf.wandb.id = str(conf.wandb.id)
        wandb.init(
            project="iBOT",
            config=OmegaConf.to_container(conf, resolve=True),  # type:ignore
            resume="allow",
            **conf.wandb,
        )
    torch.backends.cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentation(
        **conf.transform,
    )
    logging.info(f"Data augmentation: {transform}")

    pred_size = conf.model.patch_size
    dataset = ImageFolderMask(
        conf.data.path,
        transform=transform,
        patch_size=pred_size,
        pred_ratio=conf.data.pred_ratio,
        pred_ratio_var=conf.data.pred_ratio_var,
        pred_aspect_ratio=(0.3, 1 / 0.3),
        pred_shape=conf.data.pred_shape,
        pred_start_epoch=conf.data.pred_start_epoch,
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
    if conf.debug:
        _batches = []
        for i, batch in enumerate(data_loader):
            _batches.append(batch)
            if i >= 10:
                break

    logging.info(f"Data loaded: there are {len(dataset)} images.")

    vit_student, vit_teacher, cnn_student, cnn_teacher = build_models(conf)
    vit_student, vit_teacher, vit_teacher_without_ddp = setup_for_ddp(
        vit_student, vit_teacher
    )
    cnn_student, cnn_teacher, cnn_teacher_without_ddp = setup_for_ddp(
        cnn_student, cnn_teacher
    )

    teacher_without_ddp = nn.ModuleDict(
        {
            "vit": vit_teacher_without_ddp,
            "cnn": cnn_teacher_without_ddp,
        }
    )
    student = nn.ModuleDict(
        {
            "vit": vit_student,
            "cnn": cnn_student,
        }
    )
    teacher = nn.ModuleDict(
        {
            "vit": vit_teacher,
            "cnn": cnn_teacher,
        }
    )
    logging.info(
        f"Student and Teacher are built: {vit_student.__class__}, {cnn_student.__class__}"
    )

    # ============ preparing loss ... ============
    ibot_loss = iBOTLoss(
        conf.model.n_prototypes,
        conf.model.n_prototypes,
        conf.transform.global_crops_number,
        conf.transform.local_crops_number,
        conf.loss.warmup_teacher_temp,
        conf.loss.teacher_temp,
        conf.loss.warmup_teacher_patch_temp,
        conf.loss.teacher_patch_temp,
        conf.loss.warmup_teacher_temp_epochs,
        conf.epochs,
        lambda1=conf.loss.lambda1,
        lambda2=conf.loss.lambda2,
        mim_start_epoch=conf.loss.pred_start_epoch,
    ).cuda()
    _build_dino_loss = lambda: DINOLoss(
        conf.model.n_prototypes,
        ncrops=conf.transform.global_crops_number + conf.transform.local_crops_number,
        warmup_teacher_temp=conf.loss.warmup_teacher_temp,
        teacher_temp=conf.loss.teacher_temp,
        warmup_teacher_temp_epochs=conf.loss.warmup_teacher_temp_epochs,
        nepochs=conf.epochs,
    )
    cnn_dino_loss = _build_dino_loss()
    cross_dino_loss = _build_dino_loss()

    _ = ibot_loss.cuda(), cnn_dino_loss.cuda(), cross_dino_loss.cuda()
    logging.info(f"Setup losses.")

    optimizer, lr_schedulers, wd_schedulers, momentum_schedule, fp16_scaler = (
        setup_optimization(student, len(data_loader), conf)
    )

    logging.info(f"Loss, optimizer and schedulers ready.")

    # ============ setup additional evaluation ===============
    ssl_evaluator = SSLEvaluator(conf)

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0, "best_score": 0.0}
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
    elif conf.load_from is not None:
        load_path = os.path.join(conf.load_from, "checkpoint.pth")
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
    best_score = to_restore["best_score"]

    logging.info("Starting iBOT training!")
    for epoch in range(start_epoch, conf.epochs):

        data_loader.sampler.set_epoch(epoch)
        data_loader.dataset.set_epoch(epoch)

        # ======== run and log probing results ========
        if epoch % conf.evaluation.probing_freq == 0:
            probing_results = ssl_evaluator(
                vit_teacher.backbone,
                epoch - 1,
                "cuda",
                is_main_process=utils.is_main_process(),
            )
        else:
            probing_results = None
        if probing_results:
            logging.info(f"Probing results: {probing_results}")
            wandb.log({"epoch": epoch - 1, **probing_results})

            if conf.get("monitored_metric") is not None: 
                if probing_results[conf.monitored_metric] > best_score:
                    logging.info(f"New best score: {probing_results[conf.monitored_metric]}")
                    best_score = probing_results[conf.monitored_metric]
                    save_dict = {
                        "student": student.state_dict(),
                        "teacher": teacher.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch + 1,
                        "best_score": best_score,
                        "args": conf,
                        "ibot_loss": ibot_loss.state_dict(),
                    }
                    if fp16_scaler is not None:
                        save_dict["fp16_scaler"] = fp16_scaler.state_dict()
                    utils.save_on_master(save_dict, os.path.join(conf.output_dir, "best.pth"))

        # ============ training one epoch of iBOT ... ============
        logging.info(f"EPOCH {epoch}")
        train_one_epoch(
            student,
            teacher,
            teacher_without_ddp,
            ibot_loss,
            cnn_dino_loss,
            cross_dino_loss,
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
            "best_score": best_score,
            "args": conf,
            "ibot_loss": ibot_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict["fp16_scaler"] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(conf.output_dir, "checkpoint.pth"))
        if conf.saveckp_freq and ((epoch + 1) % conf.saveckp_freq == 0) and epoch:
            utils.save_on_master(
                save_dict, os.path.join(conf.output_dir, f"checkpoint{epoch:04}.pth")
            )


def train_one_epoch(
    student,
    teacher,
    teacher_without_ddp,
    vit_ibot_loss: iBOTLoss,
    cnn_dino_loss: DINOLoss,
    cross_dino_loss: DINOLoss,
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
    params_student, params_teacher = get_common_parameters(
        nn.ModuleDict({'vit': student['vit'].module, 'cnn': student['cnn'].module}), teacher_without_ddp
    )

    for it, (images, labels, masks) in enumerate(
        metric_logger.log_every(data_loader, 10, header)
    ):
        if conf.get("debug") and (it > 10):
            break

        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        lrs_and_wds_for_logging = update_optimizer(
            it, optimizer, lr_schedulers, wd_schedulers
        )

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        masks = [msk.cuda(non_blocking=True) for msk in masks]
        batch_size = len(images[0])

        num_views = (
            conf.transform.global_crops_number + conf.transform.local_crops_number
        )

        with torch.cuda.amp.autocast(fp16_scaler is not None):  # type:ignore

            # Get VISION TRANSFORMER views
            # get global views

            with torch.no_grad():
                teacher_output = teacher["vit"](
                    images[: conf.transform.global_crops_number], return_all_tokens=True
                )

            teacher_output_cls = teacher_output[:, 0, :]
            # teacher_output_register = teacher_output[:, 1, :]
            teacher_output_patch = teacher_output[:, 2:, :]

            student_output = student["vit"](
                images[: conf.transform.global_crops_number],
                mask=masks[: conf.transform.global_crops_number],
                return_all_tokens=True,
            )
            student_output_cls = student_output[:, 0, :]
            student_output_register = student_output[:, 1, :]
            student_output_patch = student_output[:, 2:, :]

            # get local views
            if len(images) > conf.transform.global_crops_number:
                student_local_view_output = student["vit"](
                    images[conf.transform.global_crops_number :], return_all_tokens=True
                )
                student_local_cls = student_local_view_output[:, 0, :]
                student_local_register = student_local_view_output[:, 1, :]
            else:
                student_local_cls = None
                student_local_register = None

            loss_vit = vit_ibot_loss(
                (student_output_cls, student_output_patch),
                (teacher_output_cls, teacher_output_patch),
                student_local_cls,
                masks,
                epoch,
            )["loss"]

            # get CNN views
            with torch.no_grad():
                cnn_teacher_output = teacher["cnn"](
                    images[: conf.transform.global_crops_number],
                    return_all_tokens=False,
                )
            cnn_student_output = student["cnn"](images, return_all_tokens=False)

            loss_cnn = cnn_dino_loss(cnn_student_output, cnn_teacher_output, epoch)

            if student_local_register is not None:
                student_output_register = torch.cat(
                    (student_output_register, student_local_register), dim=0
                )
            assert len(student_output_register) == batch_size * num_views
            loss_cross = cross_dino_loss(
                student_output_register, cnn_teacher_output, epoch
            )

            a1 = conf.loss.vit_loss_weight
            a2 = conf.loss.cnn_loss_weight
            a3 = conf.loss.cross_loss_weight
            loss = (
                (loss_vit * a1 + loss_cnn * a2 + loss_cross * a3) * 3 / (a1 + a2 + a3)
            )

            if wandb.run is not None:
                wandb.log(
                    {
                        "loss": loss.item(),
                        "vit_loss": loss_vit.item(),
                        "cnn_loss": loss_cnn.item(),
                        "cross_loss": loss_cross.item(),
                        **lrs_and_wds_for_logging,
                    }
                )

        losses_across_batch = utils.concat_all_gather(loss.view(1))
        if any([not math.isfinite(loss_.item()) for loss_ in losses_across_batch]):
            logging.info("Warning: NaN value encountered in loss")
            continue

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if conf.optimizer.clip_grad:
                param_norms = utils.clip_gradients(student, conf.optimizer.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, conf.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if conf.optimizer.clip_grad:
                fp16_scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, conf.optimizer.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, conf.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(params_student, params_teacher):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        # if param_norms is not None:
        #     metric_logger.update(param_norms=param_norms)

    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return return_dict


def build_models(conf: DictConfig):
    from src.models import get_model as get_backbone

    # student_backbone = vit_small(patch_size=16, img_size=224, n_cls_tokens=2, masked_im_modeling=conf.model.masked_im_modeling)
    # teacher_backbone = vit_small(patch_size=16, img_size=224, n_cls_tokens=2)
    student_backbone = get_backbone(
        conf.model.vit_backbone.arch,
        n_cls_tokens=2,
        masked_im_modeling=conf.model.masked_im_modeling,
        **conf.model.vit_backbone.get("kwargs", {}),
    )
    teacher_backbone = get_backbone(
        conf.model.vit_backbone.arch, n_cls_tokens=2, **conf.model.vit_backbone.get("kwargs", {})
    )
    vit_embed_dim = student_backbone.embed_dim

    # student_cnn_backbone = resnet18()
    # teacher_cnn_backbone = resnet18()
    student_cnn_backbone = get_backbone(
        conf.model.cnn_backbone.arch, **conf.model.cnn_backbone.kwargs
    )
    teacher_cnn_backbone = get_backbone(
        conf.model.cnn_backbone.arch, **conf.model.cnn_backbone.kwargs
    )
    cnn_embed_dim = student_cnn_backbone.embed_dim
    # student_cnn_backbone.fc = nn.Identity()
    # teacher_cnn_backbone.fc = nn.Identity()
    # student_cnn_backbone = ResnetWrapper(student_cnn_backbone)
    # teacher_cnn_backbone = ResnetWrapper(teacher_cnn_backbone)

    head_kw = dict(
        out_dim=conf.model.n_prototypes,
        act="gelu",
        norm_last_layer=conf.model.norm_last_layer,
    )

    student = MultiCropWrapper(
        student_backbone,
        DINOHead(
            vit_embed_dim,
            **head_kw,
        ),
    )
    teacher = MultiCropWrapper(
        teacher_backbone,
        DINOHead(
            vit_embed_dim,
            **head_kw,
        ),
    )

    cnn_student = MultiCropWrapper(
        student_cnn_backbone,
        DINOHead(
            cnn_embed_dim,
            **head_kw,
        ),
    )

    cnn_teacher = MultiCropWrapper(
        teacher_cnn_backbone,
        DINOHead(
            cnn_embed_dim,
            **head_kw,
        ),
    )

    return student, teacher, cnn_student, cnn_teacher


def setup_optimization(student, num_iters_per_epoch, conf):
    batch_size_per_gpu = conf.batch_size_per_gpu
    epochs = conf.epochs

    def compute_true_lr_from_base_lr(base_lr):
        return base_lr * batch_size_per_gpu * utils.get_world_size() / 256.0

    params_groups = []
    lr_schedulers = []
    wd_schedulers = []

    for group_name, named_parameters in get_named_parameter_groups(student, mode=conf.optimizer.get("mode")).items():

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
        conf.optimizer.momentum_teacher, 1, conf.epochs, num_iters_per_epoch
    )

    return optimizer, lr_schedulers, wd_schedulers, momentum_schedule, fp16_scaler


def get_named_parameter_groups(model, mode=None):
    """Returns a dictionary of named_parameter iterators.

    These will be used to create optimizer parameter groups.

    Args:
        model (nn.Module): model to get named parameters from
        mode (str, optional): If provided, this will affect the parameter
            selection and naming conventions. Defaults to None.
    """
    return {
        "vit": model["vit"].named_parameters(),
        "cnn": model["cnn"].named_parameters(),
    }


def get_common_parameters(model1, model2):

    # common params
    names_q, params_q, names_k, params_k = [], [], [], []
    for name_q, param_q in model1.named_parameters():
        names_q.append(name_q)
        params_q.append(param_q)
    for name_k, param_k in chain(model2.named_parameters()):
        names_k.append(name_k)
        params_k.append(param_k)
    names_common = list(set(names_q) & set(names_k))

    params_q = [
        param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common
    ]
    params_k = [
        param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common
    ]
    return params_q, params_k


def setup_for_ddp(student, teacher):
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()

    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(
            teacher, device_ids=[dist.get_rank()], broadcast_buffers=False
        )
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(
        student, device_ids=[dist.get_rank()], broadcast_buffers=False
    )

    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False

    return student, teacher, teacher_without_ddp


class SSLEvaluator:
    def __init__(self, conf):
        if conf.evaluation.do_nct_probing:
            logging.info(f"Setting up NCT probing...")
            self.nct_probe = NCTProbing(
                conf.batch_size_per_gpu, size=conf.transform.global_crops_size
            )
        else:
            self.nct_probe = None

        build_linear_probe = lambda: build_kfold_linear_probe_for_nct_patches(
            data_path=conf.nct_patches_data_path,
            input_size=conf.transform.global_crops_size,
            batch_size=conf.batch_size_per_gpu,
            device=torch.device("cuda"),
        )
        if conf.evaluation.do_clstoken_nct_probing:
            logging.info(f"Setting up patch NCT probing...")
            self.nct_clstoken_probe = build_linear_probe()
        else:
            self.nct_clstoken_probe = None

        if conf.evaluation.do_register_probing:
            logging.info(f"Setting up probing for register token")
            self.register_token_probe = build_linear_probe()
        else:
            self.register_token_probe = None

    def __call__(
        self,
        model: BackboneNetwork,
        epoch,
        device,
        is_main_process,
    ):
        logging.info(f"Running NCT Probing")
        probing_results = {}

        def extract_feature_map(model: BackboneNetwork, x: torch.Tensor):
            return model.get_feature_map(x)

        if self.nct_probe is not None:
            outputs = self.nct_probe.run_probing(
                model,
                epoch,
                "cuda",
                extract_feature_map,  # type: ignore
                is_main_process=utils.is_main_process(),
            )
            if outputs is not None:
                train_metrics, val_metrics = outputs
                self._add_metrics_to_dict(
                    probing_results, train_metrics, "train", "probing"
                )
                self._add_metrics_to_dict(
                    probing_results, val_metrics, "val", "probing"
                )

        def get_class_token(model, im):
            tokens = model(im, return_all_tokens=True)
            return tokens[:, 0, :]

        if self.nct_clstoken_probe is not None:
            outputs = self.nct_clstoken_probe.run_probing(
                model, get_class_token, is_main_process=utils.is_main_process()  # type: ignore
            )
            if outputs is not None:
                train_metrics, val_metrics = outputs
                self._add_metrics_to_dict(
                    probing_results, train_metrics, "train", "clstoken_probing_kfold"
                )
                self._add_metrics_to_dict(
                    probing_results, val_metrics, "val", "clstoken_probing_kfold"
                )

        def get_register_token(model, im):
            tokens = model(im, return_all_tokens=True)
            return tokens[:, 1, :]

        if self.register_token_probe is not None:
            outputs = self.register_token_probe.run_probing(
                model, get_register_token, is_main_process=utils.is_main_process()  # type: ignore
            )
            if outputs is not None:
                train_metrics, val_metrics = outputs
                self._add_metrics_to_dict(
                    probing_results, train_metrics, "train", "regtoken_probing_kfold"
                )
                self._add_metrics_to_dict(
                    probing_results, val_metrics, "val", "regtoken_probing_kfold"
                )

        return probing_results

    def _add_metrics_to_dict(self, d, metrics, split, name):
        d.update({f"{split}_{k}_{name}": v for k, v in metrics.items()})


class MomentumUpdater:
    def __init__(self, student, teacher, momentum_schedule):

        self.momentum_schedule = momentum_schedule

        # common params
        names_q, params_q, names_k, params_k = [], [], [], []
        for name_q, param_q in student.named_parameters():
            names_q.append(name_q)
            params_q.append(param_q)
        for name_k, param_k in teacher.named_parameters():
            names_k.append(name_k)
            params_k.append(param_k)
        names_common = list(set(names_q) & set(names_k))
        assert (
            len(names_common) > 0
        ), "No common parameters found between student and teacher - check model architectures."
        logging.info(
            f"Found {len(names_common)} common parameters between student and teacher."
        )

        params_q = [
            param_q
            for name_q, param_q in zip(names_q, params_q)
            if name_q in names_common
        ]
        params_k = [
            param_k
            for name_k, param_k in zip(names_k, params_k)
            if name_k in names_common
        ]
        self.params_q = params_q
        self.params_k = params_k

    def update(self, it):
        # EMA update for the teacher
        breakpoint()

        with torch.no_grad():
            m = self.momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(self.params_k, self.params_q):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        breakpoint()


def update_optimizer(iter, optimizer, lr_schedulers, wd_schedulers):
    # update weight decay and learning rate according to their schedule
    lrs_and_wds_for_logging = {}
    for i, (param_group, lr_schedule, wd_schedule) in enumerate(
        zip(optimizer.param_groups, lr_schedulers, wd_schedulers)
    ):
        param_group["lr"] = lr_schedule[iter]
        param_group["weight_decay"] = wd_schedule[iter]
        lrs_and_wds_for_logging.update(
            {
                f"lr_{i}": lr_schedule[iter],
                f"wd_{i}": wd_schedule[iter],
            }
        )
    return lrs_and_wds_for_logging


def get_model_from_checkpoint(checkpoint_path, model="teacher"):

    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    conf = state_dict["args"]

    def extract_state_dict_with_prefix(state_dict, prefix):
        return {
            k.replace(prefix, ""): v
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }

    vit_student, vit_teacher, cnn_student, cnn_teacher = build_models(conf)
    match model:
        case "student" | "vit_student":
            model = vit_student.backbone
            msg = model.load_state_dict(
                extract_state_dict_with_prefix(
                    state_dict["student"], "vit.module.backbone."
                )
            )
            print(msg)
        case "teacher" | "vit_teacher":
            model = vit_teacher.backbone
            msg = model.load_state_dict(
                extract_state_dict_with_prefix(state_dict["teacher"], "vit.backbone.")
            )
            print(msg)
        case "cnn_student":
            model = cnn_student.backbone
            msg = model.load_state_dict(
                extract_state_dict_with_prefix(
                    state_dict["student"], "cnn.module.backbone."
                )
            )
            print(msg)
        case "cnn_teacher":
            model = cnn_teacher.backbone
            msg = model.load_state_dict(
                extract_state_dict_with_prefix(state_dict["teacher"], "cnn.backbone.")
            )
            print(msg)
        case _:
            raise ValueError(f"Unknown model {model}")

    return model


def get_arg_parser(): 

    parser = argparse.ArgumentParser(
        "IBOT MultiModel training",
        add_help=False,
    )
    parser.add_argument(
        "--config",
        "-c",
        action=LoadOmegaConf,
        default=["conf_new/main_ibot_multimodel.yaml"],
        help="Path to one or more yaml config files.",
    )
    parser.add_argument(
        "--overrides",
        "-o",
        action=OverrideYaml,
        dest="config",
        help="Overrides for the configuration. Should be a dotlist (eg. key.subkey=value).",
        default=[],
    )
    parser.add_argument(
        "--wandb_path", type=str, default=None, help="Path to wandb run to resume."
    )
    parser.add_argument(
        "--resume_wandb",
        action="store_true",
        help="If specified, tries to resume the wandb run.",
    )
    parser.add_argument(
        "--print_cfg", action='store_true', help='Print config and exit.'
    )
    return parser


def main():
    # conf = get_conf()
    parser = argparse.ArgumentParser(
        parents=[get_arg_parser()], 
        formatter_class=ArgumentDefaultsRichHelpFormatter
    )
    args = parser.parse_args()
    conf = args.config
    if args.wandb_path is not None:
        # resume from wandb
        api = wandb.Api()
        run = api.run(args.wandb_path)
        run_conf = OmegaConf.create(run.config)
        print(f"{run.id} loaded from wandb.")
        # copy config
        conf = OmegaConf.merge(conf, run_conf)
        if args.resume_wandb:
            conf.load_from = conf.output_dir
            conf.wandb.id = run.id

    if args.print_cfg: 
        rich.print(OmegaConf.to_yaml(conf))
        sys.exit(0)

    train_ibot(conf)

    # train_ibot(conf)


if __name__ == "__main__":
    main()

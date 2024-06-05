"""Build optimizers"""
from dataclasses import dataclass
from typing import Generic, TypeVar, Protocol
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.nn import Module
import numpy as np
from .utils import LARS 
import torch 


class OptimFactory(Protocol): 
    def __call__(self, model: Module, num_epochs, num_iters_per_epoch, batch_size): 
        """Returns optimizer, lr scheduler and weight decay scheduler"""


def cosine_scheduler(
    base_value,
    final_value,
    epochs,
    niter_per_ep,
    warmup_epochs=0,
    start_warmup_value=0,
    frozen_epochs=0,
    num_annealing_phases=1,
    lr_factor_by_phase=[1.]
):
    total_iters = epochs * niter_per_ep

    schedule_phases = []

    frozen_schedule = np.array([])
    frozen_iters = frozen_epochs * niter_per_ep
    if frozen_epochs > 0:
        frozen_schedule = np.zeros(frozen_iters)
    schedule_phases.append(frozen_schedule)

    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    schedule_phases.append(warmup_schedule)

    total_iters_for_annealing = epochs * niter_per_ep - warmup_iters - frozen_iters

    for phase_idx in range(num_annealing_phases):
        factor_for_phase = lr_factor_by_phase[phase_idx]
        iters_for_phase = total_iters_for_annealing // num_annealing_phases
        iters = np.arange(iters_for_phase)
        schedule = final_value + 0.5 * (base_value * factor_for_phase - final_value) * (
            1 + np.cos(np.pi * iters / len(iters))
        )
        schedule_phases.append(schedule)

    tail = np.array([])
    total_scheduled_iters = sum([len(phase) for phase in schedule_phases])
    remaining_iters = total_iters - total_scheduled_iters
    if remaining_iters > 0: 
        tail = np.zeros(remaining_iters)
    schedule_phases.append(tail)

    schedule = np.concatenate(schedule_phases)
    assert len(schedule) == epochs * niter_per_ep
    return schedule


@dataclass
class BasicOptimizerOptions: 
    """Learning rate and weight decay"""

    base_lr: float = 0.0005
    warmup_epochs: int = 0
    frozen_epochs: int = 0
    min_lr: float = 1e-7
    weight_decay: float = 0.04  # Initial value of the weight decay
    weight_decay_end: float = 0.4  # Final value of the weight decay


def _compute_true_lr_from_base_lr(base_lr, batch_size):
    return base_lr * batch_size / 256.0


def _get_optimizer_single_param_group(named_parameters, batch_size, epochs, num_iters_per_epoch, options: BasicOptimizerOptions):
    options = options
    
    params_groups = []
    lr_schedulers = []
    wd_schedulers = []

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
            cosine_scheduler(
            _compute_true_lr_from_base_lr(
                options.base_lr,
                batch_size
            ),  # linear scaling rule
            options.min_lr,
            epochs,
            num_iters_per_epoch,
            warmup_epochs=options.warmup_epochs,
            frozen_epochs=options.frozen_epochs,
            num_annealing_phases=options.get(
                "num_annealing_phases", 1
            ),
        )
        )
    wd_schedulers.append(
        cosine_scheduler(
            options.weight_decay,
            options.weight_decay_end,
            epochs,
            num_iters_per_epoch,
        )
    )
    # second group receives 0 wd
    wd_schedulers.append(cosine_scheduler(0, 0, epochs, num_iters_per_epoch))

    return params_groups, lr_schedulers, wd_schedulers


@dataclass
class DefaultOptimizerFactory(BasicOptimizerOptions): 

    type: str = 'adamw' # type of optimizer

    def __call__(self, model: Module, num_epochs, num_iters_per_epoch, batch_size) -> tuple[Optimizer, list[list[float]], list[list[float]]]:
        named_parameters = model.named_parameters()

        param_groups, lr_sched, wd_sched = _get_optimizer_single_param_group(named_parameters, batch_size, num_epochs, num_iters_per_epoch, self)

        if self.type == "adamw":
            optimizer = torch.optim.AdamW(param_groups)  # to use with ViTs
        elif self.type == "sgd":
            optimizer = torch.optim.SGD(
                param_groups, lr=0, momentum=0.9
            )  # lr is set by scheduler
        elif self.type == "lars":
            optimizer = LARS(param_groups)  # to use with convnet and large batches

        return optimizer, lr_sched, wd_sched


def get_momentum_schedule(momentum_teacher, epochs, num_iters_per_epoch):
    momentum_schedule = cosine_scheduler(
        momentum_teacher, 1, epochs, num_iters_per_epoch
    )
    return momentum_schedule


FACTORIES = {
    'default': DefaultOptimizerFactory
}
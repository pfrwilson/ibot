from dataclasses import dataclass, field
from typing import Literal
from submitit import SlurmExecutor
from submitit.helpers import DelayedSubmission
from main_ibot_multimodel import get_conf, train_ibot
import os 


@dataclass
class VectorSlurmLauncher: 
    """Submits the function to the vector cluster to run"""

    QOS_TIMES = {
        'normal': 16, 
        'm': 12, 
        'm2': 8, 
        'm3': 4
    }

    gpus: int = 8
    partition: list[Literal['t4v2', 'a40', 'rtx6000']] = field(default_factory=lambda:['a40', 'rtx6000', 't4v2'])
    qos: Literal['normal', 'm', 'm2', 'm3'] = 'm2'
    mem_per_gpu: int = 16
    cpus_per_task: int = 4

    def get_executor(self):
        executor = SlurmExecutor(folder='.submitit', max_num_timeout=100)
        executor.update_parameters(
            gres=f'gpu:{self.gpus}',
            qos=self.qos,
            cpus_per_task=self.cpus_per_task, 
            mem=f'{self.mem_per_gpu*self.gpus}G',
            partition=",".join(self.partition), 
            signal_delay_s=60*4,
            stderr_to_stdout=True, 
            setup=["export MASTER_PORT=29500", "export MASTER_ADDR=localhost", "export TQDM_DISABLE=True", "export WANDB_RESUME=ALLOW"], 
            time=self.QOS_TIMES[self.qos] * 60,
            ntasks_per_node=self.gpus
        )
        return executor


class Main: 
    def __init__(self, conf):
        self.conf = conf

    def __call__(self): 
        os.environ['RANK'] = os.environ["SLURM_LOCALID"]
        os.environ['WORLD_SIZE'] = os.environ["SLURM_NTASKS"]

        train_ibot(self.conf)

    def checkpoint(self, *args, **kwargs): 
        return DelayedSubmission(Main(self.conf))


def main(): 
    conf = get_conf()
    launcher = VectorSlurmLauncher(
        qos='m3'
    )
    executor = launcher.get_executor()
    executor.submit(Main(conf))


if __name__ == "__main__": 
    main()
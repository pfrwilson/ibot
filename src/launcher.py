

from dataclasses import dataclass, field
import os
from pathlib import Path
from re import L
from typing import Literal
from submitit.helpers import DelayedSubmission
from submitit import SlurmExecutor
from traitlets import default


@dataclass
class Launcher: 
    """Runs the function"""

    def __call__(self, func, run_dir, *args, **kwargs): 
        return func(*args, **kwargs)


class Main: 
    def __call__(self, func, run_dir, *args, **kwargs): 
        print(run_dir)
        print(Path(run_dir).parent)
        os.makedirs(Path(run_dir).parent, exist_ok=True)
        checkpoint_dir=os.path.join(
            '/checkpoint', 
            os.environ['USER'], 
            os.environ['SLURM_JOB_ID']
        )
        if not os.path.exists(run_dir):
            try: 
                os.symlink(checkpoint_dir, run_dir, target_is_directory=True)
            except: 
                pass

        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.run_dir = run_dir
        func(*args, **kwargs)

    def checkpoint(self, *args, **kwargs): 
        return DelayedSubmission(Main(), self.func, self.run_dir, *self.args, **self.kwargs)


@dataclass
class VectorSlurmLauncher(Launcher): 
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

    def __call__(self, func, run_dir, *args, **kwargs):
        
        executor = SlurmExecutor(folder='.submitit', max_num_timeout=100)
        executor.update_parameters(
            gres=f'gpu:{self.gpus}',
            qos=self.qos,
            cpus_per_task=self.cpus_per_task, 
            mem=f'{self.mem_per_gpu*self.gpus}G',
            partition=",".join(self.partition), 
            signal_delay_s=60*4,
            stderr_to_stdout=True, 
            setup=["export MASTER_PORT=29500", "export MASTER_ADDR=localhost", "export TQDM_DISABLE=True"], 
            time=self.QOS_TIMES[self.qos] * 60,
            ntasks_per_node=self.gpus
        )
        job = executor.submit(Main(), func, run_dir, *args, **kwargs)
        print(job.job_id)


LAUNCHERS = {
    'basic': Launcher, 
    'vc': VectorSlurmLauncher
}
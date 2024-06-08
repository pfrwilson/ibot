
from argparse import ArgumentParser
import datetime
import os
import sys

import rich
from main_ibot_multimodel import get_conf


SLRM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={gpus}
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:{gpus}
#SBATCH --mem={mem}G
#SBATCH --qos=m3
#SBATCH --time=4:00:00
#SBATCH --signal=B:USR1@240
#SBATCH --open-mode=append

# Kill training process and resubmit job if it receives a SIGUSR1
handle_timeout_or_preemption() {{
  date +"%Y-%m-/%d %T"
  echo "Caught timeout or preemption signal"
  echo $(date +"%Y-%m-/%d %T") "Resubmitting job"
  scontrol requeue $SLURM_JOB_ID
  exit 0
}}
trap handle_timeout_or_preemption SIGUSR1

srun /bin/bash {bash_file} & wait
"""


SH_TEMPLATE = """echo SLURM_LOCALID:$SLURM_LOCALID
echo SLURM_PROCID:$SLURM_PROCID
echo $CUDA_VISIBLE_DEVICES

export MASTER_PORT=29500
export MASTER_ADDR=localhost
export RANK=$SLURM_LOCALID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS

WANDB_ID=$SLURM_JOB_ID
export WANDB_RUN_ID=$WANDB_ID
export WANDB_RESUME='ALLOW'
export TQDM_DISABLE='True'

python {script} {args}
"""


def main(): 
    parser = ArgumentParser(description="Make slurm script to run the program", add_help=False)
    parser.add_argument('--slrm_help', '-sh', action='help')
    parser.add_argument('--gpus', type=int, default=8)
    parser.add_argument('--job_name', default='ibot')
    parser.add_argument('--slrm_dir', default='.slurm')

    
    args, extras = parser.parse_known_args()
    rich.print(vars(args))

    conf = get_conf(extras)

    os.makedirs(args.slrm_dir, exist_ok=True)
    prefix = datetime.datetime.now().strftime('%Y-%m-%d_%H-%m-%S')

    slurm_file = os.path.join(args.slrm_dir, f"{prefix}.slrm")
    bash_file = os.path.join(args.slrm_dir, f"{prefix}.sh")

    with open(bash_file, 'w') as f: 
        f.write(SH_TEMPLATE.format(script='main_ibot_multimodel.py', args=" ".join(extras)))    
    with open(slurm_file, 'w') as f: 
        f.write(SLRM_TEMPLATE.format(
            job_name=args.job_name, 
            gpus=args.gpus, 
            mem=16*args.gpus, 
            bash_file=bash_file
        ))
    os.system(f'sbatch {slurm_file}')


if __name__ == '__main__': 
    main()
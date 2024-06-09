
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import datetime
import os
import sys

import rich
# from main_ibot_multimodel import ArgumentDefaultsRichHelpFormatter, get_conf
import importlib


SLRM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={gpus}
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:{gpus}
#SBATCH --mem={mem}G
#SBATCH --qos={qos}
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
export WANDB_RESUME='ALLOW'
export TQDM_DISABLE='True'

python {script} {args}
"""


def main(): 
    parser = ArgumentParser(description="Make slurm script to run the program", add_help=False)
    parser.add_argument('program')
    parser.add_argument('--slrm_help', '-sh', action='help')
    parser.add_argument('--slrm_gpus', type=int, default=8)
    parser.add_argument('--slrm_job_name', default='ibot')
    parser.add_argument('--slrm_dir', default='.slurm')
    parser.add_argument('--slrm_qos', default='m3')

    args, extras = parser.parse_known_args()
    rich.print(vars(args))

    program = args.program
    module = importlib.import_module(program.replace('.py', ""))
    get_parser = getattr(module, 'get_arg_parser')
    prog_parser = ArgumentParser(parents=[get_parser()], formatter_class=ArgumentDefaultsHelpFormatter)
    prog_parser.parse_args(extras)

    os.makedirs(args.slrm_dir, exist_ok=True)
    prefix = datetime.datetime.now().strftime('%Y-%m-%d_%H-%m-%S')

    slurm_file = os.path.join(args.slrm_dir, f"{prefix}.slrm")
    bash_file = os.path.join(args.slrm_dir, f"{prefix}.sh")

    with open(bash_file, 'w') as f: 
        f.write(SH_TEMPLATE.format(script=program, args=" ".join(extras)))    
    with open(slurm_file, 'w') as f: 
        f.write(SLRM_TEMPLATE.format(
            job_name=args.slrm_job_name, 
            gpus=args.slrm_gpus, 
            mem=16*args.slrm_gpus, 
            bash_file=bash_file,
            qos=args.slrm_qos,
        ))
    os.system(f'sbatch {slurm_file}')


if __name__ == '__main__': 
    main()
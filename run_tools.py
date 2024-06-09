from datetime import datetime
import os
import argparse
from omegaconf import OmegaConf


SLRM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=ibot
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

CHECKPOINT_DIR=/checkpoint/$USER/$SLURM_JOB_ID
EXPERIMENT_DIR={run_dir}

if [ $SLURM_LOCALID = 0 ]
then
    if [ ! -d $EXPERIMENT_DIR/checkpoint ] 
    then
        ln -s $CHECKPOINT_DIR $EXPERIMENT_DIR/checkpoint
    fi
fi

export MASTER_PORT=29500
export MASTER_ADDR=localhost
export RANK=$SLURM_LOCALID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS
export WANDB_RESUME='ALLOW'
export TQDM_DISABLE='True'

python {script} {args}
"""


def prepare_run(
    source_conf_path,
    experiment_dir="experiment",
    run_name="run",
    gpus=8,
    qos="m3",
    mem_per_gpu=16,
    script="train.py",
    extra_args=[],
):
    run_dir = os.path.join(experiment_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    conf_file = os.path.join(run_dir, "conf.yaml")
    source_conf = OmegaConf.load(source_conf_path)
    OmegaConf.save(source_conf, conf_file)
    bash_file = os.path.join(run_dir, "run.sh")

    with open(bash_file, "w") as f:
        f.write(
            SH_TEMPLATE.format(
                script=script,
                args=" ".join([f"-c {conf_file}"] + extra_args),
                run_dir=run_dir,
            )
        )

    slrm_file = os.path.join(run_dir, "run.slrm")
    with open(slrm_file, "w") as f:
        f.write(
            SLRM_TEMPLATE.format(
                job_name=run_name,
                gpus=gpus,
                mem=mem_per_gpu * gpus,
                qos=qos,
                bash_file=bash_file,
            )
        )

    print(f"Run dir: {run_dir}")
    print(f"SLRM file: {slrm_file}")
    print(f"Bash file: {bash_file}")
    print(f"Conf file: {conf_file}")
    print(f"Command: sbatch {slrm_file}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    prepare_run_parser = subparsers.add_parser("prepare_run")
    prepare_run_parser.add_argument("script")
    prepare_run_parser.add_argument("source_conf_path", type=str)
    prepare_run_parser.add_argument("--experiment_dir", default="experiment")
    prepare_run_parser.add_argument("--run_name", default=None)
    prepare_run_parser.add_argument("--gpus", type=int, default=8)
    prepare_run_parser.add_argument("--qos", default="m3")
    prepare_run_parser.add_argument("--mem_per_gpu", type=int, default=16)
    args, extras = parser.parse_known_args()

    if args.command == "prepare_run":
        prepare_run(
            args.source_conf_path,
            args.experiment_dir,
            args.run_name,
            args.gpus,
            args.qos,
            args.mem_per_gpu,
            args.script,
            extras,
        )


if __name__ == "__main__":
    main()

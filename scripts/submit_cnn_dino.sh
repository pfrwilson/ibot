#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --output=slurm_logs/pretrain%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:8
#SBATCH --mem=128G
#SBATCH --qos=m
#SBATCH --time=8:00:00
#SBATCH --signal=B:USR1@240
#SBATCH --open-mode=append

echo RUNNING

# Kill training process and resubmit job if it receives a SIGUSR1
handle_timeout_or_preemption() {
  date +"%Y-%m-%d %T"
  echo "Caught timeout or preemption signal"
  echo $(date +"%Y-%m-%d %T") "Resubmitting job"
  scontrol requeue $SLURM_JOB_ID
  exit 0
}
trap handle_timeout_or_preemption SIGUSR1



srun /bin/bash slurm/run_cnn_dino.sh & wait
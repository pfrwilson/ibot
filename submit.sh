#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --output=%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:8
#SBATCH --mem=128G
#SBATCH --qos=m3
#SBATCH --time=4:00:00
#SBATCH --signal=B:USR1@240
#SBATCH --open-mode=append

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <program>"
    exit 1
fi
echo Running program $1

# Kill training process and resubmit job if it receives a SIGUSR1
handle_timeout_or_preemption() {
  date +"%Y-%m-%d %T"
  echo "Caught timeout or preemption signal"
  echo $(date +"%Y-%m-%d %T") "Resubmitting job"
  scontrol requeue $SLURM_JOB_ID
  exit 0
}
trap handle_timeout_or_preemption SIGUSR1

srun /bin/bash $1 & wait
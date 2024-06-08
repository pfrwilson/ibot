#!/bin/bash
#!/bin/bash
#SBATCH --job-name=multiple-nodes-multiple-gpus
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=2
#SBATCH --mem=64gb
#SBATCH --output=imagenet.%j.out
#SBATCH --error=imagenet.%j.err
#SBATCH --wait-all-nodes=1

export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

export MASTER_ADDR="$(hostname --fqdn)"
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"

for index in $(seq 0 $(($SLURM_NTASKS-1))); do 
    /opt/slurm/bin/srun -lN$index --mem=64G --gres=gpu:2 -c $SLURM_CPUS_ON_NODE -N 1 -n 1 -r $index bash -c "hostname" &
done


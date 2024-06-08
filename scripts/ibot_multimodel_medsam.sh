echo SLURM_LOCALID:$SLURM_LOCALID
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

IMAGENET_DATA=/scratch/ssd004/datasets/imagenet256/train
MICROUS_DATA=/ssd005/projects/exactvu_pca/unlabelled_microus_png

python main_ibot_multimodel.py -c conf_new/main_ibot_multimodel_medsam.yaml

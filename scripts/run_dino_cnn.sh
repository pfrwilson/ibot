echo SLURM_LOCALID:$SLURM_LOCALID
echo SLURM_PROCID:$SLURM_PROCID
echo $CUDA_VISIBLE_DEVICES

CHECKPOINT_DIR=/checkpoint/$USER/$SLURM_JOB_ID
EXPERIMENT_DIR=experiment/$SLURM_JOB_ID

if [ $SLURM_LOCALID = 0 ]
then
    if [ ! -d $EXPERIMENT_DIR ] 
    then
        ln -s $CHECKPOINT_DIR $EXPERIMENT_DIR
    fi
fi

export MASTER_PORT=29501
export MASTER_ADDR=localhost
export RANK=$SLURM_LOCALID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS

export WANDB_RUN_ID=$SLURM_JOB_ID
export WANDB_RESUME='ALLOW'
export TQDM_DISABLE='1'

IMAGENET_DATA=/scratch/ssd004/datasets/imagenet256/train 
MICROUS_DATA=/ssd005/projects/exactvu_pca/unlabelled_microus_png

python main_dino.py \
    --output_dir $EXPERIMENT_DIR \
    --initial_crop_size 768 \
    --local_crops_number 4 \
    --global_crops_scale 0.5 1 \
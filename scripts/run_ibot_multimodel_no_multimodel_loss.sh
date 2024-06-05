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

# link checkpoint dir to experiments 

export MASTER_PORT=29500
export MASTER_ADDR=localhost
export RANK=$SLURM_LOCALID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS

export WANDB_RUN_ID=$SLURM_JOB_ID
export WANDB_RESUME='ALLOW'
export WANDB_NAME='ibot-multimodel-control'
export TQDM_DISABLE='True'


IMAGENET_DATA=/scratch/ssd004/datasets/imagenet256/train 
MICROUS_DATA=/ssd005/projects/exactvu_pca/unlabelled_microus_png

python main_ibot_multimodel.py -c conf_new/main_ibot_multimodel.yaml --overrides output_dir=$EXPERIMENT_DIR loss.cnn_loss_weight=0 loss.cross_loss_weight=0
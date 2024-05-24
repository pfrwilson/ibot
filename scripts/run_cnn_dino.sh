echo SLURM_LOCALID:$SLURM_LOCALID
echo SLURM_PROCID:$SLURM_PROCID
echo $CUDA_VISIBLE_DEVICES

CHECKPOINT_DIR=/checkpoint/$USER/$SLURM_JOB_ID
export MASTER_PORT=29501
export MASTER_ADDR=localhost
export RANK=$SLURM_LOCALID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS

export WANDB_RUN_ID=$SLURM_JOB_ID
export WANDB_RESUME='ALLOW'


IMAGENET_DATA=/scratch/ssd004/datasets/imagenet256/train 
MICROUS_DATA=/ssd005/projects/exactvu_pca/unlabelled_microus_png

python main_ibot.py \
    --data_path=$IMAGENET_DATA \
    --batch_size_per_gpu 32 \
    --output_dir $CHECKPOINT_DIR \
    --global_crops_size 512 \
    --arch resnet10t \
    --momentum_teacher .9995 \
    --clip_grad 1 \
    --lr 1e-4 \
    --epochs 100 \
    --use_masked_im_modeling False \
    --patch_size 32 \
    --do_unsupervised_eval
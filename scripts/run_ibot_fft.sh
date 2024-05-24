echo SLURM_LOCALID:$SLURM_LOCALID
echo SLURM_PROCID:$SLURM_PROCID
echo $CUDA_VISIBLE_DEVICES

CHECKPOINT_DIR=/checkpoint/$USER/$SLURM_JOB_ID
export MASTER_PORT=29500
export MASTER_ADDR=localhost
export RANK=$SLURM_LOCALID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS

export WANDB_RUN_ID=$SLURM_JOB_ID
export WANDB_RESUME='ALLOW'
export WANDB_NAME='ibot_fft'
export TQDM_DISABLE='True'

IMAGENET_DATA=/scratch/ssd004/datasets/imagenet256/train 
MICROUS_DATA=/ssd005/projects/exactvu_pca/unlabelled_microus_png

python main_ibot.py \
    --arch vit_small_fft \
    --data_path=$MICROUS_DATA \
    --do_nct_probing \
    --batch_size_per_gpu 1 \
    --output_dir $CHECKPOINT_DIR \
    --patch_size 32 \
    --global_crops_size 512 \
    --initial_resize_size 1024 \
    --initial_crop_size 1024 \
    --global_crops_scale 0.6 0.6 \
    --blur_prob_1 0 \
    --blur_prob_2 0 \
    --solarization_prob 0 \
    --jitter_prob 0 \
    --momentum_teacher .9995 \
    --clip_grad 1 \
    --lr 1e-4 \
    --epochs 200 \
    --do_nct_probing
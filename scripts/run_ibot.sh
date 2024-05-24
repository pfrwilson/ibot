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
export TQDM_DISABLE='True'

IMAGENET_DATA=/scratch/ssd004/datasets/imagenet256/train 
MICROUS_DATA=/ssd005/projects/exactvu_pca/unlabelled_microus_png

python main_ibot.py \
    --data_path=$MICROUS_DATA \
    --do_nct_probing \
    --batch_size_per_gpu 1 \
    --output_dir $CHECKPOINT_DIR \
    --global_crops_size 512 \
    --arch medsam \
    --momentum_teacher .9995 \
    --clip_grad 1 \
    --lr 1e-4 \
    --epochs 200 \
    --load_model_from "/checkpoint/pwilson/12540874/checkpoint.pth" 
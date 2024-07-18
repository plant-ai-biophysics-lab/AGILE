#! /bin/bash -l
#SBATCH --job-name=paibl
#SBATCH --output=outputs/run_%A_%a.out
#SBATCH --partition=gpum
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=ewranario@ucdavis.edu
#SBATCH --time=72:00:00 # Change the time accordingly
#SBATCH --mail-type=ALL
#SBATCH --error=outputs/run_%A_%a.err
#SBATCH --cpus-per-task=8

source ~/.bashrc
conda activate lightning
unset PYTHONPATH
WORKDIR="/group/jmearlesgrp/scratch/eranario/Active-Deep-Learning/scripts/active-learning"
cd $WORKDIR

# logging args
DATASET="pascal"
METHOD="entropy"
NAME="${DATASET}_${METHOD}_no-ckpt_0718"

# define script args
ROOT_DIR="/group/jmearlesgrp/data/PASCAL-VOC-2012"
LOGS_DIR="/group/jmearlesgrp/intermediate_data/eranario/Active-Learning/PASCAL_logs/$METHOD/$NAME"
BATCH_SIZE=24
LR=0.0002
EPOCHS=100
IMAGE_SIZE=512
ROUNDS=20
CHUNK=0.02

# make logs dir if it doesnt exist
mkdir -p $LOGS_DIR

# Run the Python training script
python pascal_detector.py \
    --root_dir $ROOT_DIR \
    --logs_dir $LOGS_DIR \
    --name $NAME \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --epochs $EPOCHS \
    --image_size $IMAGE_SIZE \
    --method $METHOD \
    --rounds $ROUNDS \
    --chunk $CHUNK

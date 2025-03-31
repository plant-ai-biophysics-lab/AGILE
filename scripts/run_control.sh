#!/bin/bash

CURRENT_PATH=$(pwd)/../

# Define the arguments
DATA="grape"
TRANSFER="syn2day"
CHECKPOINT="checkpoints/control_sd15_ini.ckpt"  # Path to pretrained model
MODEL_CONFIG="$CURRENT_PATH/models/cldm_v15.yaml"  # Path to model config file
SOURCE_IMAGES_PATH="/group/jmearlesgrp/scratch/eranario/AGILE/datasets/grape_detection_syntheticday/reformatted/train/images"  # Path to source images
TARGET_IMAGES_PATH="/group/jmearlesgrp/scratch/eranario/AGILE/datasets/grape_detection_californiaday/reformatted/train/images"  # Path to target images
BATCH_SIZE=1  # Batch size for training
IMAGE_SIZE=512  # Input image size
EPOCHS=30  # Number of epochs for training
PARAM="eps"  # Parameterization for loss
SPREAD_FACTOR=2.0  # Spread factor for Gaussian map
CONTROL_STRENGTH=1.0  # Control strength for ControlNet
PROMPT_EMBEDDING="" # Path to optimized embeddings
BETAS='[[20, 50]]'  # Betas for ControlNet
UGS="5.0"

# Make logs directory
RUN_NAME="CONTROL_${DATA}_${TRANSFER}_strength-${CONTROL_STRENGTH}_ugs-${UGS}_sf-${SPREAD_FACTOR}_${DESCRIPTION}"
LOGS_DIR="logs/$RUN_NAME"  # Directory to save logs
mkdir -p $LOGS_DIR

# Run the Python script with the arguments
python main.py \
  --model_config $MODEL_CONFIG \
  --checkpoint $CHECKPOINT \
  --source_images_path $SOURCE_IMAGES_PATH \
  --target_images_path $TARGET_IMAGES_PATH \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --logs_dir $LOGS_DIR \
  --control_strength $CONTROL_STRENGTH \
  --betas "$BETAS" \
  --image_size $IMAGE_SIZE \
  --spread_factor $SPREAD_FACTOR \
  --param $PARAM \
  --unconditional_guidance_scale $UGS \
  --use_transforms \
  --control_attentions \
  --prompt_embedding $PROMPT_EMBEDDING \
  # --subset

#!/bin/bash

CURRENT_PATH=$(pwd)/../

# Define the arguments
CHECKPOINT="/data2/eranario/intermediate_data/AGILE/control_sd15_ini.ckpt"  # Path to pretrained model
MODEL_CONFIG="$CURRENT_PATH/models/cldm_v15.yaml"  # Path to model config file
SD_LOCKED=true  # Stable diffusion decoder locked
ONLY_MID_CONTROL=false  # Only mid control of ControlNet
SOURCE_IMAGES_PATH="/data2/eranario/data/AGILE-Datasets/Grape-Detection/Synthetic/images"  # Path to source images
TARGET_IMAGES_PATH="/data2/eranario/data/AGILE-Datasets/Grape-Detection/BordenDayRow/train/images"  # Path to target images
BATCH_SIZE=1  # Batch size for training
IMAGE_SIZE=512  # Input image size
EPOCHS=10  # Number of epochs for training
PARAM="eps"  # Parameterization for loss
OPTIMIZE_EMBEDDINGS=true  # Optimize embeddings
PROMPT="grape"  # Prompt for text embeddings
OPTIMIZE_EPOCHS=1  # Number of epochs for optimizing embeddings
OPTIMIZE_STEPS=50  # Number of optimization steps
SPREAD_FACTOR=3.0  # Spread factor for Gaussian map

# Make logs directory
RUN_NAME="1216_OPTIMIZE_SF-3"
LOGS_DIR="/data2/eranario/intermediate_data/AGILE/borden_syn2day/$RUN_NAME"  # Directory to save logs
mkdir -p $LOGS_DIR

# Run the Python script with the arguments
python3 $CURRENT_PATH/main.py \
  --checkpoint "$CHECKPOINT" \
  --model_config "$MODEL_CONFIG" \
  $( [ "$SD_LOCKED" = true ] && echo "--sd_locked" ) \
  --only_mid_control "$ONLY_MID_CONTROL" \
  --source_images_path "$SOURCE_IMAGES_PATH" \
  --target_images_path "$TARGET_IMAGES_PATH" \
  --batch_size "$BATCH_SIZE" \
  --image_size "$IMAGE_SIZE" \
  --epochs "$EPOCHS" \
  --logs_dir "$LOGS_DIR" \
  --param "$PARAM" \
  $( [ "$OPTIMIZE_EMBEDDINGS" = true ] && echo "--optimize_embeddings" ) \
  --prompt "$PROMPT" \
  --optimize_epochs "$OPTIMIZE_EPOCHS" \
  --optimize_steps "$OPTIMIZE_STEPS" \
  --spread_factor "$SPREAD_FACTOR"

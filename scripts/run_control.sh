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
EPOCHS=50  # Number of epochs for training
PARAM="eps_attn"  # Parameterization for loss
OPTIMIZE_EPOCHS=1  # Number of epochs for optimizing embeddings
OPTIMIZE_STEPS=50  # Number of optimization steps
SPREAD_FACTOR=4.0  # Spread factor for Gaussian map
CONTROL_STRENGTH=5.0  # Control strength for ControlNet
LR=0.00001  # Learning rate
PROMPT_EMBEDDING="/data2/eranario/intermediate_data/AGILE/borden_syn2day/1216_OPTIMIZE_SF-3_EPOCHS-20_PART-3/text_optimizer/optimized_embeddings.pt" # Path to optimized embeddings
BETAS="[[35, 20], [35, 10], [35, 5], [35, 1]]"  # Betas for ControlNet

# Make logs directory
RUN_NAME="1220_CONTROL_GUIDE-TRAIN_LAST-BETA"
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
  --lr "$LR" \
  --control_strength "$CONTROL_STRENGTH" \
  --spread_factor "$SPREAD_FACTOR" \
  --prompt_embedding "$PROMPT_EMBEDDING" \
  --control_attentions \
  --betas "$BETAS"

# AGILE
**A**ttention-**G**uided **I**mage and **L**abel Translation for **E**fficient Cross-Domain Plant Trait Identification

## 1. Prepare pretrained weights

Go to official ControlNet repository to obtain pretrained weights: https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md

## 2. Create a new conda environment

```
conda env create -n agile python=3.10
pip install -r requirements.txt
```

## 3. Run Text Optimization

Edit these lines in `scripts/run_optimize.sh`:

```
CHECKPOINT="./control_sd15_ini.ckpt"  # Path to pretrained model
RUN_NAME="INSERT-CUSTOM-RUNNAME-HERE"
LOGS_DIR="../$RUN_NAME"  # Directory to save logs
```

Change `CHECKPOINT` to the path where the pretrained weights are in step 1.
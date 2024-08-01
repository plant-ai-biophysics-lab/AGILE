import argparse

from pathlib import Path

from src.util import create_model

def main(args):
    
    # create model from config
    model = create_model(args.model_config).cpu()
    print(model)
    
if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="Path to pretrained model (stable diffusion with controlnet).")
    ap.add_argument("--model_config", type=Path, required=True,
                    help="Path to model config file (yaml file in models folder).")
    args = ap.parse_args()
    
    main(args)
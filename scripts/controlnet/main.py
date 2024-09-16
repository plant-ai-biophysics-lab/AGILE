import argparse
import os
import torch
import wandb
import random
import pytorch_lightning as pl

from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from src.util import create_model, load_state_dict, PermuteTransform, initialize_weights
from src.dataset import ControlNetDataset
from src.logger import ImageLogger
from src.model import TextEmbeddingOptimizer
from lightning.pytorch.loggers import WandbLogger

def main(args):
    
    # create model from config and load from chkpt
    mismatch_count = 0
    total_count = 0
    model = create_model(args.model_config).cpu()
    if args.checkpoint:
        state_dict = load_state_dict(args.checkpoint, location='cpu')
        for name, param in model.named_parameters():
            total_count += 1
            if name in state_dict:
                if state_dict[name].shape != param.shape:
                    mismatch_count += 1
                    initialize_weights(param.data)
                    del state_dict[name]
        model.load_state_dict(state_dict, strict=False)
        print(f"Mismatches: {mismatch_count} out of {total_count}")
    
    # edit model parameters
    model.learning_rate = args.learning_rate
    model.sd_locked = args.sd_locked
    model.only_mid_control = args.only_mid_control
    model.parameterization = args.param
    model.attn_loss_weight = 2.0
    
    strength = 1.0
    model.control_scales = ([strength] * 13)
    
    print(f"Using parameterization: {model.parameterization}")
    
    # get optimized prompt embedding if exists
    if args.prompt_embedding is not None:
        prompt = torch.load(args.prompt_embedding)
    else:
        prompt = args.prompt
    
    # prepare dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        PermuteTransform()
    ])
    dataset = ControlNetDataset(
        source_images_path=args.source_images_path,
        target_images_path=args.target_images_path,
        prompt=prompt,
        transform=transform
    )
    dataloader = DataLoader(
        dataset,
        num_workers=0,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    logger = ImageLogger(epoch_frequency=args.logger_freq)
    logger.train_dataloader = dataloader
    
    # prepare wandb logger
    wandb_logger = WandbLogger(
        entity='paibl',
        project='controlnet',
        name=f"{args.logs_dir.name}_initial_training",
        save_dir=args.logs_dir,
    )
    
    # start training
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        default_root_dir=args.logs_dir,
        precision = 32,
        callbacks = [logger],
        logger=wandb_logger,
        accumulate_grad_batches=args.batch_size*4,
    )
    trainer.fit(model, dataloader)
    
    # end wandb
    wandb.finish()
    
    # optimize embeddings
    if args.optimize_embeddings:
                
        # Set up the Wandb logger for embedding optimization
        wandb.init(
            entity='paibl',
            project='controlnet',
            name=f"{args.logs_dir.name}_embedding_optimization",
            dir=args.logs_dir,
            resume=False
        )
        
        # Initialize Text Embedding Optimizer
        text_optimizer = TextEmbeddingOptimizer(
            prompt=prompt,
            model=model,
            batch_size=args.batch_size,
            lr=0.01,
            ddim_steps=50,
            unconditional_guidance_scale=20.0,
            logs_dir=os.path.join(args.logs_dir, "text_optimizer"),
            optimization_steps=args.optimize_steps
        )
        
        # Use subset of dataloader
        num_indices = min(1, len(dataset))
        subset_indices = random.sample(range(len(dataset)), num_indices)
        print(f"Optimizing embeddings images: {subset_indices}")
        subset_dataset = Subset(dataset, subset_indices)
        dataloader = DataLoader(
            subset_dataset,
            num_workers=0,
            batch_size=args.batch_size,
            shuffle=True
        )
        
        # Initialize Trainer for embedding optimization
        text_optimizer.train(dataloader, num_epochs=args.optimize_epochs)
        
if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Path to pretrained model (stable diffusion with controlnet).")
    ap.add_argument("--model_config", type=Path, required=True,
                    help="Path to model config file (yaml file in models folder).")
    ap.add_argument("--lr", type=float, dest="learning_rate", default=1e-5,
                    help = "Learning rate for the model.")
    ap.add_argument("--sd_locked", action="store_true",
                    help="If set, stable diffusion decoder will be locked.")
    ap.add_argument("--only_mid_control", type=bool, default=False,
                    help="If set, only mid control of ControlNet will be used.")
    ap.add_argument("--source_images_path", type=Path, required=True,
                    help="Path to source images.")
    ap.add_argument("--target_images_path", type=Path, required=True,
                    help="Path to target images.")
    ap.add_argument("--batch_size", type=int, default=1,
                    help="Batch size for training.")
    ap.add_argument("--logger_freq", type=int, default=1,
                    help="Logging frequency.")
    ap.add_argument("--epochs", type=int, default=100,
                    help="Number of epochs for training.")
    ap.add_argument("--logs_dir", type=Path, default="logs",
                    help="Directory to save logs.")
    ap.add_argument("--param", type=str, default="eps",
                    help="Parameterization for calculation loss: x0, eps, v, or eps_attn")
    ap.add_argument("--optimize_embeddings", action="store_true",
                    help="If set, embeddings will be optimized.")
    ap.add_argument("--prompt", type=str, default="grape",
                    help="Prompt for text embeddings.")
    ap.add_argument("--optimize_epochs", type=int, default=1,
                    help="Number of epochs for optimizing embeddings.")
    ap.add_argument("--optimize_steps", type=int, default=100,
                    help="Number of optimization steps.")
    ap.add_argument("--prompt_embedding", type=Path, default=None,
                    help="Path to optimized prompt embedding.")
    # TODO: Add args for strength and unconditional guidance scale
    # TODO: Add args to generate final images
    args = ap.parse_args()
    
    main(args)
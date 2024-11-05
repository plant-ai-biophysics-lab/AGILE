import argparse
import os
import torch
import wandb
import random
import ast
import pytorch_lightning as pl

from pathlib import Path
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from src.util import create_model, load_state_dict, PermuteTransform, initialize_weights
from src.dataset import ControlNetDataset
from src.logger import ImageLogger
from src.model import TextEmbeddingOptimizer, AttentionGuidance
from lightning.pytorch.loggers import WandbLogger

def main(args):
    
    #######################################################
    ################## MODEL SETUP ########################
    #######################################################
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
    
    strength = args.control_strength
    model.control_scales = ([strength] * 13)
    
    print(f"Using parameterization: {model.parameterization}")
    
    #######################################################
    ################## TRAINING SETUP #####################
    #######################################################
    
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
        transform=transform,
        optimizing=args.optimize_embeddings
    )
    dataloader = DataLoader(
        dataset,
        num_workers=0,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # logger = ImageLogger(epoch_frequency=args.logger_freq, disabled=args.generate_images)
    # logger.train_dataloader = dataloader
    
    # # prepare wandb logger
    # wandb_logger = WandbLogger(
    #     entity='paibl',
    #     project='controlnet',
    #     name=f"{args.logs_dir.name}_initial_training",
    #     save_dir=args.logs_dir,
    # )
    
    # # start training
    # trainer = pl.Trainer(
    #     max_epochs=args.epochs,
    #     default_root_dir=args.logs_dir,
    #     precision = 32,
    #     callbacks = [logger],
    #     logger=wandb_logger,
    #     accumulate_grad_batches=args.batch_size*4,
    # )
    # trainer.fit(model, dataloader)
    
    # # end wandb
    # wandb.finish()
    
    #######################################################
    ############### EMBEDDING OPTIMIZATION ################
    #######################################################
    
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
        num_indices = min(3, len(dataset))
        subset_indices = random.sample(range(len(dataset)), num_indices)
        print(f"Optimizing embeddings images: {subset_indices}")
        subset_dataset = Subset(dataset, subset_indices)
        optimize_dataloader = DataLoader(
            subset_dataset,
            num_workers=0,
            batch_size=args.batch_size,
            shuffle=True
        )
        
        # Initialize Trainer for embedding optimization
        text_optimizer.train(optimize_dataloader, num_epochs=args.optimize_epochs)
        wandb.finish()
        
    #######################################################
    ################# ATTENTION GUIDANCE ##################
    #######################################################
    
    if args.control_attentions:
        
        # TODO: REMOVE THIS DEBUG AFTER DRAFT RUNS
        # reduce dataset size to 5 for debugging
        dataset = Subset(dataset, random.sample(range(len(dataset)), 5))
        dataloader_debug = DataLoader(
            dataset,
            num_workers=0,
            batch_size=args.batch_size,
            shuffle=True
        )
        
        # Set up the Wandb logger for embedding optimization
        wandb.init(
            entity='paibl',
            project='controlnet',
            name=f"{args.logs_dir.name}_attention_guidance",
            dir=args.logs_dir,
            resume=False
        )
        
        # get beta pairs
        betas = ast.literal_eval(args.betas)
        
        # Initialize Attention Guidance
        attention_guidance = AttentionGuidance(
            prompt=prompt,
            model=model,
            batch_size=args.batch_size,
            ddim_steps=50,
            unconditional_guidance_scale=20.0,
            logs_dir=os.path.join(args.logs_dir, "attention_guidance"),
            betas=betas
        )
        
        attention_guidance.train(dataloader_debug, num_epochs=args.optimize_epochs)

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
    ap.add_argument("--prompt", type=str, default="None",
                    help="Prompt for text embeddings.")
    ap.add_argument("--optimize_epochs", type=int, default=1,
                    help="Number of epochs for optimizing embeddings.")
    ap.add_argument("--optimize_steps", type=int, default=100,
                    help="Number of optimization steps.")
    ap.add_argument("--prompt_embedding", type=Path, default=None,
                    help="Path to optimized prompt embedding.")
    ap.add_argument("--control_attentions", action="store_true",
                    help="If set, control attentions will be used.")
    ap.add_argument("--control_strength", type=float, default=1.0,
                    help="Strength of control.")
    ap.add_argument("--generate_images", action="store_true",
                    help="If set, final images will be generated at the end.")
    ap.add_argument('--betas', type=str, required=True, 
                    help="List of beta pairs, e.g., '[[50, 40], [50, 25], [50, 20]]'")
    args = ap.parse_args()
    
    main(args)
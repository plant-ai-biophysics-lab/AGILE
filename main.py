import argparse
import os
import torch
import random
import ast
import pytorch_lightning as pl

from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from src.util import create_model, load_state_dict, PermuteTransform, initialize_weights
from src.dataset import ControlNetDataset
from src.logger import ImageLogger
from src.model import TextEmbeddingOptimizer, AttentionGuidance
from src.util import calculate_metrics

# hide warnings
import warnings
warnings.filterwarnings("ignore")

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
    print(f"Using learning rate: {args.learning_rate}")
    model.learning_rate = args.learning_rate
    print(f"Using sd_locked: {args.sd_locked}")
    model.sd_locked = args.sd_locked
    print(f"Using only_mid_control: {args.only_mid_control}")
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
        print("Using optimized embeddings!")
        prompt = torch.load(args.prompt_embedding)
        # squeeze prompt if 3 dims
        if prompt.dim() == 3:
            prompt = prompt.squeeze(0)
    elif args.prompt == "None":
        prompt = torch.randn(77, 768)
    else:
        prompt = args.prompt
    
    # prepare dataset and dataloader
    betas = ast.literal_eval(args.betas)
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        PermuteTransform()
    ])
    dataset = ControlNetDataset(
        source_images_path=args.source_images_path,
        target_images_path=args.target_images_path,
        prompt=prompt,
        transform=transform,
        optimizing=args.optimize_embeddings,
        spread_factor=args.spread_factor,
        betas=betas,
        img_size=args.image_size,
        use_transforms=args.use_transforms
    )
    dataloader = DataLoader(
        dataset,
        num_workers=0,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    if not args.subset:
        
        logger = ImageLogger(epoch_frequency=args.logger_freq, disabled=args.generate_images)
        logger.train_dataloader = dataloader
        
        # start training
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            default_root_dir=args.logs_dir,
            precision = 32,
            callbacks = [logger],
            accumulate_grad_batches=args.batch_size*4
        )
        trainer.fit(model, dataloader)
    
    #######################################################
    ############### EMBEDDING OPTIMIZATION ################
    #######################################################
    
    # optimize embeddings
    if args.optimize_embeddings:

        # Initialize Text Embedding Optimizer
        text_optimizer = TextEmbeddingOptimizer(
            prompt=prompt,
            model=model,
            batch_size=args.batch_size,
            lr=0.01,
            ddim_steps=50,
            image_size=args.image_size,
            unconditional_guidance_scale=args.unconditional_guidance_scale,
            timestep_to_optimize=f"timestep_{args.timestep}",
            logs_dir=os.path.join(args.logs_dir, "text_optimizer"),
            optimization_steps=args.optimize_steps,
        )
        
        # Use subset of dataloader
        num_indices = min(5, len(dataset))
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
        
    #######################################################
    ################# ATTENTION GUIDANCE ##################
    #######################################################
    
    if args.control_attentions:
        
        finetune_steps = 1
        generated = None
        
        model.only_mid_control = args.only_mid_control
        strength = args.control_strength
        model.control_scales = ([strength] * 13)
        
        for step in range(finetune_steps):
            
            print(f"Finetuning step: {step}")
            dataset_step = ControlNetDataset(
                source_images_path=args.source_images_path,
                target_images_path=args.source_images_path,
                prompt=prompt,
                transform=transform,
                optimizing=args.optimize_embeddings,
                spread_factor=args.spread_factor,
                betas=betas,
                generated=generated,
                apply_bbox_mask=args.mask_bbox,
            )
            dataloader_step = DataLoader(
                dataset_step,
                num_workers=0,
                batch_size=args.batch_size,
                shuffle=True
            )
        
            # get beta pairs
            betas = ast.literal_eval(args.betas)
            
            # Initialize Attention Guidance
            generated = os.path.join(args.logs_dir, f"attention_guidance_{step}")
            attention_guidance = AttentionGuidance(
                prompt=prompt,
                model=model,
                batch_size=args.batch_size,
                ddim_steps=50,
                unconditional_guidance_scale=args.unconditional_guidance_scale,
                logs_dir=generated,
                betas=betas,
                resize_final=args.resize_final
            )
            
            attention_guidance.train(dataloader_step, original_size=dataset.source_image_size, target_size=dataset.target_images_size)
            
        #######################################################
        ################# METRICS CALCULATION #################
        #######################################################
        
        metrics = calculate_metrics(
            real_path=args.target_images_path,
            generated_path=generated,
            output_dir=args.logs_dir
        )
        print("Metrics saved to:", os.path.join(args.logs_dir, "metrics.txt"))
        print("Metrics:", metrics)

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
    ap.add_argument("--image_size", type=int, default=512,
                    help="Input image size.")
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
    ap.add_argument("--control_strength", type=float, default=5.0,
                    help="Strength of control.")
    ap.add_argument("--generate_images", action="store_true",
                    help="If set, final images will be generated at the end.")
    ap.add_argument('--betas', type=str, default='[[50, 40], [50, 30], [50, 25]]', 
                    help="List of beta pairs, e.g., '[[50, 40], [50, 30], [50, 25]]'")
    ap.add_argument('--spread_factor', type=float, default=4.0,
                    help="Spread factor for Gaussian map, for larger objects, recommend 2.")
    ap.add_argument('--timestep', type=int, default=30,
                    help="Timestep in backward process to optimize text embedding.")
    ap.add_argument('--unconditional_guidance_scale', type=float, default=5.0,
                    help="Scale for unconditional guidance.")
    ap.add_argument('--use_transforms', action='store_true',
                    help="If set, transforms will be used for data augmentation.")
    ap.add_argument("--resize_final", action="store_true",
                    help="If set, final images will be resized to target size.")
    ap.add_argument("--subset", action="store_true",
                    help="If set, skip training.")
    ap.add_argument("--mask_bbox", action="store_true",
                    help="If set, mask bounding box will be used.")
    args = ap.parse_args()
    
    main(args)
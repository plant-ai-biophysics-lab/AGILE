#!/usr/bin/env python3
"""
This combined script does two things:
  1. Prepares the dataset by scanning the specified source directory (source_dir) and target directory (target_dir):
     Splits the images into training and testing sets and creates symlinks in:
       dest_dir/dataset_name/
         ├── train_A/    (Source training images)
         ├── train_B/    (Target training images)
         ├── test_A/     (Source test images)
         ├── test_B/     (Target test images)
         ├── fixed_prompt_a.txt
         └── fixed_prompt_b.txt

  2. Immediately launches training of a CycleGAN Turbo model using the prepared dataset.
  
Usage example:
    python train.py \
      --source_dir /path/to/source \
      --target_dir="/path/to/target" \
      --dest_dir /path/to/data \
      --dataset_name my_dataset \
      --train_img_prep "resize_286_randomcrop_256x256_hflip" --val_img_prep "no_resize" \
      --output_dir "./output" \
      --tracker_project_name "project_name"\
      --max_train_steps=number_of_steps\
   
If using Accelerate:
    accelerate config 

    export NCCL_P2P_DISABLE=1

    accelerate launch --main_process_port 29501 train.py \
     --source_dir /path/to/source \
     --target_dir="/path/to/target" \
     --dest_dir /path/to/data \
     --dataset_name my_dataset \
     --pretrained_model_name_or_path="stabilityai/sd-turbo" \
     --train_img_prep "resize_286_randomcrop_256x256_hflip" --val_img_prep "no_resize" \
     --learning_rate="1e-5" --max_train_steps=number_of_steps \
     --train_batch_size=1 --gradient_accumulation_steps=1 \
     --enable_xformers_memory_efficient_attention --validation_steps 250 \
     --lambda_gan 0.5 --lambda_idt 1 --lambda_cycle 1 \
     --output_dir="/path/to/output" \
     --tracker_project_name="my_project" \
      
Any training arguments (such as --output_dir, --max_train_steps, etc.) are parsed via your existing 
parse_args_unpaired_training() utility.
"""

import os
import gc
import copy
import random
from glob import glob
import argparse
import sys

import lpips
import torch
import wandb
import numpy as np 
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel
from diffusers.optimization import get_scheduler
from peft.utils import get_peft_model_state_dict
from cleanfid.fid import get_folder_features, build_feature_extractor, frechet_distance
import vision_aided_loss

# Local module imports – ensure these are available in your PYTHONPATH
from cgt_utils.model_utils import make_1step_sched
from cgt_utils.cyclegan_turbo import CycleGAN_Turbo, VAE_encode, VAE_decode, initialize_unet, initialize_vae
from cgt_utils.training_utils import UnpairedDataset, build_transform, parse_args_unpaired_training
from cgt_utils.dino_struct import DinoStructureLoss

# from model_utils import make_1step_sched
# from cyclegan_turbo import CycleGAN_Turbo, VAE_encode, VAE_decode, initialize_unet, initialize_vae
# from training_utils import UnpairedDataset, build_transform, parse_args_unpaired_training
# from dino_struct import DinoStructureLoss


###############################################################################
# Dataset Preparation Functions
###############################################################################

def collect_images(source_dir, target_dir):
    """
    Traverse source_dir and target_dir and collect image paths.
    Returns two lists:
      - src_images: images from source_dir
      - tgt_images: images from target_dir
    """
    src_images = []
    tgt_images = []
    
    if os.path.isdir(source_dir):
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif"):
            src_images.extend(glob(os.path.join(source_dir, ext)))
    if os.path.isdir(target_dir):
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif"):
            tgt_images.extend(glob(os.path.join(target_dir, ext)))
    return src_images, tgt_images

def split_dataset(file_list, train_ratio):
    """
    Shuffle and split a list of file paths into train and test sets.
    Returns (train_files, test_files)
    """
    random.shuffle(file_list)
    split_index = int(len(file_list) * train_ratio)
    return file_list[:split_index], file_list[split_index:]

def symlink_files(file_list, dest_dir):
    """
    Create symlinks in dest_dir with sequential zero-padded filenames.
    """
    os.makedirs(dest_dir, exist_ok=True)
    for idx, src in enumerate(sorted(file_list)):
        filename = f"{idx:06d}.png"  # e.g. 000000.png, 000001.png, etc.
        dst = os.path.join(dest_dir, filename)
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)

def create_fixed_prompts(dest_dir):
    """
    Create fixed_prompt_a.txt and fixed_prompt_b.txt if they do not exist.
    """
    prompt_a = os.path.join(dest_dir, "fixed_prompt_a.txt")
    prompt_b = os.path.join(dest_dir, "fixed_prompt_b.txt")
    if not os.path.exists(prompt_a):
        with open(prompt_a, "w") as f:
            f.write("")
    if not os.path.exists(prompt_b):
        with open(prompt_b, "w") as f:
            f.write("")

def prepare_dataset(prep_args):
    """
    Prepare the dataset based on the provided source and destination arguments.
    Creates the following structure:
       dest_dir/dataset_name/
         ├── train_A/
         ├── train_B/
         ├── test_A/
         ├── test_B/
         ├── fixed_prompt_a.txt
         └── fixed_prompt_b.txt
    """
    dataset_dir = os.path.join(prep_args.dest_dir, prep_args.dataset_name)
    for sub in ["train_A", "train_B", "test_A", "test_B"]:
        os.makedirs(os.path.join(dataset_dir, sub), exist_ok=True)
    
    src_images, tgt_images = collect_images(prep_args.source_dir, prep_args.target_dir)
    print(f"Collected {len(src_images)} source images and {len(tgt_images)} target images.")
    if not src_images or not tgt_images:
        raise ValueError("No images found – please check your source directory structure.")
    
    train_src, test_src = split_dataset(src_images, prep_args.train_ratio)
    train_tgt, test_tgt = split_dataset(tgt_images, prep_args.train_ratio)
    print(f"Source train/test split: {len(train_src)} / {len(test_src)}")
    print(f"Target train/test split: {len(train_tgt)} / {len(test_tgt)}")
    
    symlink_files(train_src, os.path.join(dataset_dir, "train_A"))
    symlink_files(train_tgt, os.path.join(dataset_dir, "train_B"))
    symlink_files(test_src, os.path.join(dataset_dir, "test_A"))
    symlink_files(test_tgt, os.path.join(dataset_dir, "test_B"))
    create_fixed_prompts(dataset_dir)
    print(f"Dataset prepared at: {dataset_dir}")
    return dataset_dir

def cleanup_symlinks(dataset_dir):
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            filepath = os.path.join(root, file)
            if os.path.islink(filepath):
                os.remove(filepath)
    print("Removed all symlinks from:", dataset_dir)

###############################################################################
# Training Function (CycleGAN Turbo)
###############################################################################
def train(training_args):
    accelerator = Accelerator(gradient_accumulation_steps=training_args.gradient_accumulation_steps,
                                log_with=training_args.report_to)
    set_seed(training_args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(training_args.output_dir, "checkpoints"), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo",
                                                subfolder="tokenizer",
                                                revision=training_args.revision,
                                                use_fast=False)
    noise_scheduler_1step = make_1step_sched()
    text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo",
                                                 subfolder="text_encoder").cuda()

    unet, l_modules_unet_encoder, l_modules_unet_decoder, l_modules_unet_others = initialize_unet(
        training_args.lora_rank_unet, return_lora_module_names=True)
    vae_a2b, vae_lora_target_modules = initialize_vae(training_args.lora_rank_vae,
                                                      return_lora_module_names=True)

    weight_dtype = torch.float32
    vae_a2b.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)

    if training_args.gan_disc_type == "vagan_clip":
        net_disc_a = vision_aided_loss.Discriminator(cv_type='clip',
                                                     loss_type=training_args.gan_loss_type,
                                                     device="cuda")
        net_disc_a.cv_ensemble.requires_grad_(False)
        net_disc_b = vision_aided_loss.Discriminator(cv_type='clip',
                                                     loss_type=training_args.gan_loss_type,
                                                     device="cuda")
        net_disc_b.cv_ensemble.requires_grad_(False)

    crit_cycle, crit_idt = torch.nn.L1Loss(), torch.nn.L1Loss()

    if training_args.enable_xformers_memory_efficient_attention:
        unet.enable_xformers_memory_efficient_attention()
    if training_args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    if training_args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    unet.conv_in.requires_grad_(True)
    vae_b2a = copy.deepcopy(vae_a2b)
    params_gen = CycleGAN_Turbo.get_traininable_params(unet, vae_a2b, vae_b2a)

    vae_enc = VAE_encode(vae_a2b, vae_b2a=vae_b2a)
    vae_dec = VAE_decode(vae_a2b, vae_b2a=vae_b2a)

    optimizer_gen = torch.optim.AdamW(params_gen,
                                        lr=training_args.learning_rate,
                                        betas=(training_args.adam_beta1, training_args.adam_beta2),
                                        weight_decay=training_args.adam_weight_decay,
                                        eps=training_args.adam_epsilon)
    params_disc = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
    optimizer_disc = torch.optim.AdamW(params_disc,
                                         lr=training_args.learning_rate,
                                         betas=(training_args.adam_beta1, training_args.adam_beta2),
                                         weight_decay=training_args.adam_weight_decay,
                                         eps=training_args.adam_epsilon)

    dataset_train = UnpairedDataset(dataset_folder=training_args.dataset_folder,
                                    image_prep=training_args.train_img_prep,
                                    split="train",
                                    tokenizer=tokenizer)
    train_dataloader = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=training_args.train_batch_size,
                                                   shuffle=True,
                                                   num_workers=training_args.dataloader_num_workers)
    T_val = build_transform(training_args.val_img_prep)
    fixed_caption_src = dataset_train.fixed_caption_src
    fixed_caption_tgt = dataset_train.fixed_caption_tgt

    l_images_src_test = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_src_test.extend(glob(os.path.join(training_args.dataset_folder, "test_A", ext)))
    l_images_tgt_test = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_tgt_test.extend(glob(os.path.join(training_args.dataset_folder, "test_B", ext)))
    l_images_src_test, l_images_tgt_test = sorted(l_images_src_test), sorted(l_images_tgt_test)

    if accelerator.is_main_process:
        feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)
        """
        FID reference statistics for A -> B translation
        """
        output_dir_ref = os.path.join(training_args.output_dir, "fid_reference_a2b")
        os.makedirs(output_dir_ref, exist_ok=True)
        # transform all images according to the validation transform and save them
        for _path in tqdm(l_images_tgt_test):
            _img = T_val(Image.open(_path).convert("RGB"))
            outf = os.path.join(output_dir_ref, os.path.basename(_path)).replace(".jpg", ".png")
            if not os.path.exists(outf):
                _img.save(outf)
        # compute the features for the reference images
        ref_features = get_folder_features(output_dir_ref, model=feat_model, num_workers=0, num=None,
                        shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                        mode="clean", custom_fn_resize=None, description="", verbose=True,
                        custom_image_tranform=None)
        a2b_ref_mu, a2b_ref_sigma = np.mean(ref_features, axis=0), np.cov(ref_features, rowvar=False)
        """
        FID reference statistics for B -> A translation
        """
        # transform all images according to the validation transform and save them
        output_dir_ref = os.path.join(training_args.output_dir, "fid_reference_b2a")
        os.makedirs(output_dir_ref, exist_ok=True)
        for _path in tqdm(l_images_src_test):
            _img = T_val(Image.open(_path).convert("RGB"))
            outf = os.path.join(output_dir_ref, os.path.basename(_path)).replace(".jpg", ".png")
            if not os.path.exists(outf):
                _img.save(outf)
        # compute the features for the reference images
        ref_features = get_folder_features(output_dir_ref, model=feat_model, num_workers=0, num=None,
                        shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                        mode="clean", custom_fn_resize=None, description="", verbose=True,
                        custom_image_tranform=None)
        b2a_ref_mu, b2a_ref_sigma = np.mean(ref_features, axis=0), np.cov(ref_features, rowvar=False)

    lr_scheduler_gen = get_scheduler(training_args.lr_scheduler, optimizer=optimizer_gen,
        num_warmup_steps=training_args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=training_args.max_train_steps * accelerator.num_processes,
        num_cycles=training_args.lr_num_cycles, power=training_args.lr_power)
    lr_scheduler_disc = get_scheduler(training_args.lr_scheduler, optimizer=optimizer_disc,
        num_warmup_steps=training_args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=training_args.max_train_steps * accelerator.num_processes,
        num_cycles=training_args.lr_num_cycles, power=training_args.lr_power)

    net_lpips = lpips.LPIPS(net='vgg')
    net_lpips.cuda()
    net_lpips.requires_grad_(False)

    fixed_a2b_tokens = tokenizer(fixed_caption_tgt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
    fixed_a2b_emb_base = text_encoder(fixed_a2b_tokens.cuda().unsqueeze(0))[0].detach()
    fixed_b2a_tokens = tokenizer(fixed_caption_src, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
    fixed_b2a_emb_base = text_encoder(fixed_b2a_tokens.cuda().unsqueeze(0))[0].detach()
    del text_encoder, tokenizer  # free up some memory

    unet, vae_enc, vae_dec, net_disc_a, net_disc_b = accelerator.prepare(unet, vae_enc, vae_dec, net_disc_a, net_disc_b)
    net_lpips, optimizer_gen, optimizer_disc, train_dataloader, lr_scheduler_gen, lr_scheduler_disc = accelerator.prepare(
        net_lpips, optimizer_gen, optimizer_disc, train_dataloader, lr_scheduler_gen, lr_scheduler_disc
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(training_args.tracker_project_name, config=dict(vars(training_args)))

    first_epoch = 0
    global_step = 0
    progress_bar = tqdm(range(0, training_args.max_train_steps), initial=global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,)
    # turn off eff. attn for the disc
    for name, module in net_disc_a.named_modules():
        if "attn" in name:
            module.fused_attn = False
    for name, module in net_disc_b.named_modules():
        if "attn" in name:
            module.fused_attn = False

    for epoch in range(first_epoch, training_args.max_train_epochs):
        for step, batch in enumerate(train_dataloader):
            l_acc = [unet, net_disc_a, net_disc_b, vae_enc, vae_dec]
            with accelerator.accumulate(*l_acc):
                img_a = batch["pixel_values_src"].to(dtype=weight_dtype)
                img_b = batch["pixel_values_tgt"].to(dtype=weight_dtype)

                bsz = img_a.shape[0]
                fixed_a2b_emb = fixed_a2b_emb_base.repeat(bsz, 1, 1).to(dtype=weight_dtype)
                fixed_b2a_emb = fixed_b2a_emb_base.repeat(bsz, 1, 1).to(dtype=weight_dtype)
                timesteps = torch.tensor([noise_scheduler_1step.config.num_train_timesteps - 1] * bsz, device=img_a.device).long()

                """
                Cycle Objective
                """
                # A -> fake B -> rec A
                cyc_fake_b = CycleGAN_Turbo.forward_with_networks(img_a, "a2b", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_a2b_emb)
                cyc_rec_a = CycleGAN_Turbo.forward_with_networks(cyc_fake_b, "b2a", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_b2a_emb)
                loss_cycle_a = crit_cycle(cyc_rec_a, img_a) * training_args.lambda_cycle
                loss_cycle_a += net_lpips(cyc_rec_a, img_a).mean() * training_args.lambda_cycle_lpips
                # B -> fake A -> rec B
                cyc_fake_a = CycleGAN_Turbo.forward_with_networks(img_b, "b2a", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_b2a_emb)
                cyc_rec_b = CycleGAN_Turbo.forward_with_networks(cyc_fake_a, "a2b", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_a2b_emb)
                loss_cycle_b = crit_cycle(cyc_rec_b, img_b) * training_args.lambda_cycle
                loss_cycle_b += net_lpips(cyc_rec_b, img_b).mean() * training_args.lambda_cycle_lpips
                accelerator.backward(loss_cycle_a + loss_cycle_b, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, training_args.max_grad_norm)
    
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                """
                Generator Objective (GAN) for task a->b and b->a (fake inputs)
                """
                fake_a = CycleGAN_Turbo.forward_with_networks(img_b, "b2a", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_b2a_emb)
                fake_b = CycleGAN_Turbo.forward_with_networks(img_a, "a2b", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_a2b_emb)
                loss_gan_a = net_disc_a(fake_b, for_G=True).mean() * training_args.lambda_gan
                loss_gan_b = net_disc_b(fake_a, for_G=True).mean() * training_args.lambda_gan
                accelerator.backward(loss_gan_a + loss_gan_b, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, training_args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()
                optimizer_disc.zero_grad()

                """
                Identity Objective
                """
                idt_a = CycleGAN_Turbo.forward_with_networks(img_b, "a2b", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_a2b_emb)
                loss_idt_a = crit_idt(idt_a, img_b) * training_args.lambda_idt
                loss_idt_a += net_lpips(idt_a, img_b).mean() * training_args.lambda_idt_lpips
                idt_b = CycleGAN_Turbo.forward_with_networks(img_a, "b2a", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_b2a_emb)
                loss_idt_b = crit_idt(idt_b, img_a) * training_args.lambda_idt
                loss_idt_b += net_lpips(idt_b, img_a).mean() * training_args.lambda_idt_lpips
                loss_g_idt = loss_idt_a + loss_idt_b
                accelerator.backward(loss_g_idt, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, training_args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                """
                Discriminator for task a->b and b->a (fake inputs)
                """
                loss_D_A_fake = net_disc_a(fake_b.detach(), for_real=False).mean() * training_args.lambda_gan
                loss_D_B_fake = net_disc_b(fake_a.detach(), for_real=False).mean() * training_args.lambda_gan
                loss_D_fake = (loss_D_A_fake + loss_D_B_fake) * 0.5
                accelerator.backward(loss_D_fake, retain_graph=False)
                if accelerator.sync_gradients:
                    params_to_clip = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, training_args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

                """
                Discriminator for task a->b and b->a (real inputs)
                """
                loss_D_A_real = net_disc_a(img_b, for_real=True).mean() * training_args.lambda_gan
                loss_D_B_real = net_disc_b(img_a, for_real=True).mean() * training_args.lambda_gan
                loss_D_real = (loss_D_A_real + loss_D_B_real) * 0.5
                accelerator.backward(loss_D_real, retain_graph=False)
                if accelerator.sync_gradients:
                    params_to_clip = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, training_args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

            logs = {}
            logs["cycle_a"] = loss_cycle_a.detach().item()
            logs["cycle_b"] = loss_cycle_b.detach().item()
            logs["gan_a"] = loss_gan_a.detach().item()
            logs["gan_b"] = loss_gan_b.detach().item()
            logs["disc_a"] = loss_D_A_fake.detach().item() + loss_D_A_real.detach().item()
            logs["disc_b"] = loss_D_B_fake.detach().item() + loss_D_B_real.detach().item()
            logs["idt_a"] = loss_idt_a.detach().item()
            logs["idt_b"] = loss_idt_b.detach().item()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    eval_unet = accelerator.unwrap_model(unet)
                    eval_vae_enc = accelerator.unwrap_model(vae_enc)
                    eval_vae_dec = accelerator.unwrap_model(vae_dec)
                    if global_step % training_args.viz_freq == 1:
                        for tracker in accelerator.trackers:
                            if tracker.name == "wandb":
                                viz_img_a = batch["pixel_values_src"].to(dtype=weight_dtype)
                                viz_img_b = batch["pixel_values_tgt"].to(dtype=weight_dtype)
                                log_dict = {
                                    "train/real_a": [wandb.Image(viz_img_a[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)],
                                    "train/real_b": [wandb.Image(viz_img_b[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)],
                                }
                                log_dict["train/rec_a"] = [wandb.Image(cyc_rec_a[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)]
                                log_dict["train/rec_b"] = [wandb.Image(cyc_rec_b[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)]
                                log_dict["train/fake_b"] = [wandb.Image(fake_b[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)]
                                log_dict["train/fake_a"] = [wandb.Image(fake_a[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)]
                                tracker.log(log_dict)
                                gc.collect()
                                torch.cuda.empty_cache()

                    if global_step % training_args.checkpointing_steps == 1:
                        outf = os.path.join(training_args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        sd = {}
                        sd["l_target_modules_encoder"] = l_modules_unet_encoder
                        sd["l_target_modules_decoder"] = l_modules_unet_decoder
                        sd["l_modules_others"] = l_modules_unet_others
                        sd["rank_unet"] = training_args.lora_rank_unet
                        sd["sd_encoder"] = get_peft_model_state_dict(eval_unet, adapter_name="default_encoder")
                        sd["sd_decoder"] = get_peft_model_state_dict(eval_unet, adapter_name="default_decoder")
                        sd["sd_other"] = get_peft_model_state_dict(eval_unet, adapter_name="default_others")
                        sd["rank_vae"] = training_args.lora_rank_vae
                        sd["vae_lora_target_modules"] = vae_lora_target_modules
                        sd["sd_vae_enc"] = eval_vae_enc.state_dict()
                        sd["sd_vae_dec"] = eval_vae_dec.state_dict()
                        torch.save(sd, outf)
                        gc.collect()
                        torch.cuda.empty_cache()

                    # compute val FID and DINO-Struct scores
                    if global_step % training_args.validation_steps == 1:
                        _timesteps = torch.tensor([noise_scheduler_1step.config.num_train_timesteps - 1] * 1, device="cuda").long()
                        net_dino = DinoStructureLoss()
                        """
                        Evaluate "A->B"
                        """
                        fid_output_dir = os.path.join(training_args.output_dir, f"fid-{global_step}/samples_a2b")
                        os.makedirs(fid_output_dir, exist_ok=True)
                        l_dino_scores_a2b = []
                        # get val input images from domain a
                        for idx, input_img_path in enumerate(tqdm(l_images_src_test)):
                            if idx > training_args.validation_num_images and training_args.validation_num_images > 0:
                                break
                            outf = os.path.join(fid_output_dir, f"{idx}.png")
                            with torch.no_grad():
                                input_img = T_val(Image.open(input_img_path).convert("RGB"))
                                img_a = transforms.ToTensor()(input_img)
                                img_a = transforms.Normalize([0.5], [0.5])(img_a).unsqueeze(0).cuda()
                                eval_fake_b = CycleGAN_Turbo.forward_with_networks(img_a, "a2b", eval_vae_enc, eval_unet,
                                    eval_vae_dec, noise_scheduler_1step, _timesteps, fixed_a2b_emb[0:1])
                                eval_fake_b_pil = transforms.ToPILImage()(eval_fake_b[0] * 0.5 + 0.5)
                                eval_fake_b_pil.save(outf)
                                a = net_dino.preprocess(input_img).unsqueeze(0).cuda()
                                b = net_dino.preprocess(eval_fake_b_pil).unsqueeze(0).cuda()
                                dino_ssim = net_dino.calculate_global_ssim_loss(a, b).item()
                                l_dino_scores_a2b.append(dino_ssim)
                        dino_score_a2b = np.mean(l_dino_scores_a2b)
                        gen_features = get_folder_features(fid_output_dir, model=feat_model, num_workers=0, num=None,
                            shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                            mode="clean", custom_fn_resize=None, description="", verbose=True,
                            custom_image_tranform=None)
                        ed_mu, ed_sigma = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
                        score_fid_a2b = frechet_distance(a2b_ref_mu, a2b_ref_sigma, ed_mu, ed_sigma)
                        print(f"step={global_step}, fid(a2b)={score_fid_a2b:.2f}, dino(a2b)={dino_score_a2b:.3f}")

                        """
                        compute FID for "B->A"
                        """
                        fid_output_dir = os.path.join(training_args.output_dir, f"fid-{global_step}/samples_b2a")
                        os.makedirs(fid_output_dir, exist_ok=True)
                        l_dino_scores_b2a = []
                        # get val input images from domain b
                        for idx, input_img_path in enumerate(tqdm(l_images_tgt_test)):
                            if idx > training_args.validation_num_images and training_args.validation_num_images > 0:
                                break
                            outf = os.path.join(fid_output_dir, f"{idx}.png")
                            with torch.no_grad():
                                input_img = T_val(Image.open(input_img_path).convert("RGB"))
                                img_b = transforms.ToTensor()(input_img)
                                img_b = transforms.Normalize([0.5], [0.5])(img_b).unsqueeze(0).cuda()
                                eval_fake_a = CycleGAN_Turbo.forward_with_networks(img_b, "b2a", eval_vae_enc, eval_unet,
                                    eval_vae_dec, noise_scheduler_1step, _timesteps, fixed_b2a_emb[0:1])
                                eval_fake_a_pil = transforms.ToPILImage()(eval_fake_a[0] * 0.5 + 0.5)
                                eval_fake_a_pil.save(outf)
                                a = net_dino.preprocess(input_img).unsqueeze(0).cuda()
                                b = net_dino.preprocess(eval_fake_a_pil).unsqueeze(0).cuda()
                                dino_ssim = net_dino.calculate_global_ssim_loss(a, b).item()
                                l_dino_scores_b2a.append(dino_ssim)
                        dino_score_b2a = np.mean(l_dino_scores_b2a)
                        gen_features = get_folder_features(fid_output_dir, model=feat_model, num_workers=0, num=None,
                            shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                            mode="clean", custom_fn_resize=None, description="", verbose=True,
                            custom_image_tranform=None)
                        ed_mu, ed_sigma = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
                        score_fid_b2a = frechet_distance(b2a_ref_mu, b2a_ref_sigma, ed_mu, ed_sigma)
                        print(f"step={global_step}, fid(b2a)={score_fid_b2a}, dino(b2a)={dino_score_b2a:.3f}")
                        logs["val/fid_a2b"], logs["val/fid_b2a"] = score_fid_a2b, score_fid_b2a
                        logs["val/dino_struct_a2b"], logs["val/dino_struct_b2a"] = dino_score_a2b, dino_score_b2a
                        del net_dino  # free up memory

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= training_args.max_train_steps:
                break

if __name__ == "__main__":
    # First, parse dataset preparation arguments and training arguments.
    # Dataset preparation arguments:
    prep_parser = argparse.ArgumentParser(description="Combined prepare-and-train for CycleGAN Turbo.")
    prep_parser.add_argument("--source_dir", type=str, required=True,
                             help="Path to the source directory.")
    prep_parser.add_argument("--target_dir", type=str, required=True,
                             help="Path to the target directory.")
    prep_parser.add_argument("--dest_dir", type=str, required=True,
                             help="Destination directory where the dataset will be created.")
    prep_parser.add_argument("--dataset_name", type=str, required=True,
                             help="Name for the dataset folder (will be created under dest_dir).")
    prep_parser.add_argument("--train_ratio", type=float, default=0.8,
                             help="Proportion of images to use for training (default: 0.8).")
    # Let remaining training arguments pass through.
    args, remaining = prep_parser.parse_known_args()
    argv = sys.argv
    # Prepare the dataset; the prepared dataset folder will be:
    #   dest_dir/dataset_name
    sys.argv = [sys.argv[0]]
    dataset_folder = prepare_dataset(args)
    create_fixed_prompts(args.dest_dir)  
    
    # Now parse with the updated remaining args
    sys.argv.append("--dataset_folder=" + dataset_folder)

    # Add output_dir if not specified in remaining
    if not any("--output_dir=" in arg for arg in remaining):
        sys.argv.append("--output_dir=" + os.path.join(args.dest_dir, "output"))

    # Add tracker_project_name if not specified in remaining
    if not any("--tracker_project_name=" in arg for arg in remaining):
        sys.argv.append("--tracker_project_name=" + args.dataset_name)

    # Now add back all the remaining arguments
    sys.argv.extend(remaining)

    training_args = parse_args_unpaired_training()
    # training_args.dataset_folder = dataset_folder
    # Start training.
    train(training_args)
    
    cleanup_symlinks(training_args.dataset_folder)

